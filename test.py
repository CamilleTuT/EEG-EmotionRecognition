import torch
import numpy as np
import torch.nn.functional as F
from models.GNNLSTM import GNNLSTM
from DEAPDataset_Spacial import DEAPDataset, train_val_test_split
from torch_geometric.loader import DataLoader
# import matplotlib.pyplot as plt
# import itertools
np.set_printoptions(precision=2)


def map_to_category(value, thresholds):
    """
  value: float, 预测值
  thresholds: list, 阈值列表，用于将预测值映射到类别标签
    """
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return i
    return len(thresholds)


# 定义函数将多个情感维度的预测结果映射为多标签
def map_to_multilabel(predictions, thresholds_per_dimension):
    """
  predictions: list, 包含每个情感维度的预测值
  thresholds_per_dimension: list of lists, 每个情感维度的阈值列表
  """
    multilabel = []
    for pred, thresholds in zip(predictions, thresholds_per_dimension):
        label = map_to_category(pred, thresholds)
        multilabel.append(label)
    return multilabel


def test(args):
    ROOT_DIR = 'D:/Myworks/Learn/Research/EEG'
    RAW_DIR = '/DEAP/data_preprocessed_matlab'
    PROCESSED_DIR = '/202083270302ywh/GCN-LSTM-deap/ers'
    dataset = DEAPDataset(root=ROOT_DIR, raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR,
                          participant_from=args.participant_from, participant_to=args.participant_to)
    # 5 testing samples per participant (30/5/5)
    _, _, test_set = train_val_test_split(dataset)

    test_loader = DataLoader(test_set, batch_size=1)

    # MODEL PARAMETERS
    in_channels = test_set[0].num_node_features

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    data = dataset[0].to(device)
    print(data)
    # Instantiate models
    targets = ['valence', 'arousal', 'dominance', 'liking'][:args.n_targets]
    models = [GNNLSTM(in_channels, hidden_channels=64, target=target).to(device).eval() for target in targets]

    # Load best performing params on validation
    for i in range(len(targets)):
        models[i].load_state_dict(torch.load(f'./best_params_{i}'))

    mses = []
    i = 1
    p_1 = []
    p_2 = []
    p_3 = []
    p_4 = []
    t_1 = []
    t_2 = []
    t_3 = []
    t_4 = []
    thresholds = [5.5]
    # DataBatch(x=[32, 7680], edge_index=[2, 396], edge_attr=[396], y=[1, 4], batch=[32], ptr=[2])
    for batch in test_loader:
        batch = batch.to(device)
        predictions = [model(batch, visualize_convolutions=False) for model in models]
        predictions = torch.stack(predictions, dim=1).squeeze()
        # tensor([4.4144, 4.4492, 5.0555, 3.9407], device='cuda:0',  grad_fn= < SqueezeBackward0 >)
        print('-Predictions-')
        print(predictions.cpu().detach().numpy(), '\n')
        print('-Ground truth-')
        i = i + 1
        print(batch.y.cpu().detach().numpy(), '\n')
        mse = F.mse_loss(predictions, batch.y.narrow(1, 0, len(targets))).item()
        mses.append(mse)
        print(f'Mean average error: {F.l1_loss(predictions, batch.y.narrow(1, 0, len(targets))).item()}')
        print(f'Mean squared error: {mse}')
        p_1.append(predictions[0])
        p_2.append(predictions[1])
        p_3.append(predictions[2])
        p_4.append(predictions[3])

        t_1.append(batch.y[0][0])
        t_2.append(batch.y[0][1])
        t_3.append(batch.y[0][2])
        t_4.append(batch.y[0][3])
    # 将预测值转换为类别标签
    val_labels_p = [map_to_category(pred, thresholds) for pred in p_1]
    aro_labels_p = [map_to_category(pred, thresholds) for pred in p_2]
    dom_labels_p = [map_to_category(pred, thresholds) for pred in p_3]
    lik_labels_p = [map_to_category(pred, thresholds) for pred in p_4]

    # 将真实标签转换为多标签
    val_labels_t = [map_to_category(truth, thresholds) for truth in t_1]
    aro_labels_t = [map_to_category(truth, thresholds) for truth in t_2]
    dom_labels_t = [map_to_category(truth, thresholds) for truth in t_3]
    lik_labels_t = [map_to_category(truth, thresholds) for truth in t_4]

    true_multilabels = [[map_to_category(val, thresholds), map_to_category(aro, thresholds),
                         map_to_category(dom, thresholds), map_to_category(lik, thresholds)]
                        for val, aro, dom, lik in zip(t_1, t_2, t_3, t_4)]

    total_samples = len(t_1)

    # valence
    # correct_val_predictions = 0
    # for true_val, predict_val in zip(val_labels_t, val_labels_p):
    #     if true_val == predict_val:
    #         correct_val_predictions += 1
    # val_accuracy = correct_val_predictions / total_samples
    # print("valence Accuracy:", val_accuracy)

    # arousal
    correct_aro_predictions = 0
    for true_val, predict_val in zip(aro_labels_t, aro_labels_p):
        if true_val == predict_val:
            correct_aro_predictions += 1
    aro_accuracy = correct_aro_predictions / total_samples
    print("arousal Accuracy:", aro_accuracy)

    # valence & arousal
    # correct_val_aro_predictions = 0
    # for true_val, predict_val, true_aro, predict_aro in zip(val_labels_t, val_labels_p, aro_labels_t, aro_labels_p):
    #     if true_val == predict_val and true_aro == predict_aro:
    #         correct_val_aro_predictions += 1
    # val_aro_accuracy = correct_val_aro_predictions / total_samples
    # print("valence & arousal Accuracy:", val_aro_accuracy)

    p_1 = [t.cpu().detach().numpy() for t in p_1]
    g_1 = [t.cpu().detach().numpy() for t in t_1]
    p_2 = [t.cpu().detach().numpy() for t in p_2]
    g_2 = [t.cpu().detach().numpy() for t in t_2]
    p_3 = [t.cpu().detach().numpy() for t in p_3]
    g_3 = [t.cpu().detach().numpy() for t in t_3]
    p_4 = [t.cpu().detach().numpy() for t in p_4]
    g_4 = [t.cpu().detach().numpy() for t in t_4]
    # p_1 = p_1.cpu().detach().numpy()
    # g_1 = g_1.cpu().detach().numpy()

    print('----------------')
    print(f'MEAN SQUARED ERROR FOR TEST SET: {np.array(mses).mean()}')
