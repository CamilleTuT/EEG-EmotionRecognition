import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from DEAPDataset_freq import DEAPDatasetEEGFeatures, train_val_test_split
from models.GatedGraphConvGRU import GatedGraphConvGRU
from torch.utils.tensorboard import SummaryWriter


# Define training and eval functions for each epoch (will be shared for all models)
def train_epoch(kf_count, target_num, target_name, model, loader, optim, criterion, device, writer, total_train_step):
    model.train()
    model.training = True
    epoch_losses = []
    for batch in tqdm(loader):
        optim.zero_grad()
        batch = batch.to(device)
        target = (batch.y[:, target_num] > 5).float()
        output = model(batch).squeeze(1)
        loss = criterion(output, target)
        loss.backward()
        optim.step()
        total_train_step += 1
        writer.add_scalar(f"Train Loss_{target_name}_{kf_count}", loss.item(), total_train_step)
    return total_train_step
    # for batch in tqdm(loader):
    #     batch = batch.to(device)
    #     target = batch.y.T[model.target].unsqueeze(1)
    #     output = model(batch)
    #     mse_loss = criterion(output, target)
    #     # REGULARIZATION
    #     l1_regularization = torch.tensor(0, dtype=torch.float).to(device)
    #     # l2_regularization = torch.tensor(0, dtype=torch.float).to(device)
    #     for param in model.parameters():
    #         l1_regularization += (torch.norm(param, 1) ** 2).float()
    #         # l2_regularization += (torch.norm(param, 2)**2).float()
    #     loss = mse_loss + 0.02 * l1_regularization
    #     optim.zero_grad()
    #     loss.backward()
    #     optim.step()
    #     total_train_step += target.shape[0]
    #     if total_train_step % 64 == 0:
    #         print("Train Batch Counter: {}, Loss: {}".format(total_train_step, mse_loss.item()))
    #         writer.add_scalar(f"Train Loss_{target_name}", mse_loss.item(), total_train_step)
    #
    #     epoch_losses.append(mse_loss.item())
    # epoch_mean_loss = np.array(epoch_losses).mean()
    # model.train_losses.append(epoch_mean_loss)


def eval_epoch(kf_count, target_num, target_name, model, loader, criterion, device, epoch, writer, total_val_step, model_is_training):
    model.eval()
    model.training = False
    val_count = 0
    val_loss = 0
    val_acc = 0
    # Evaluation
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target = (batch.y[:, target_num] > 5).float()
            output = model(batch).squeeze(1)
            # print(output)
            loss = criterion(output, target)
            val_loss += loss.item()
            total_val_step += 1
            val_acc += torch.eq(output > 0.5, target > 0.5).sum().item()
            val_count += target.shape[0]

        # for batch in loader:
        #     batch = batch.to(device)
        #     output = model(batch)
        #     print(output)
        #     target = batch.y.T[model.target].unsqueeze(1)
        #     batch_size = target.shape[0]
        #     batch_count = 0
        #     loss = criterion(output, target)
        #     val_loss += loss.item()
        #     total_val_step += 1
        #     while batch_count < batch_size:
        #         if output[batch_count][0] < 5.5 and target[batch_count][0] < 5.5:
        #             val_acc += 1
        #         elif output[batch_count][0] > 5.5 and target[batch_count][0] > 5.5:
        #             val_acc += 1
        #         batch_count += 1
        #         val_count += 1

        print("Loss on the validation set: {}".format(val_loss))
        print("Accuracy on the validation set: {}".format(val_acc/val_count))
        writer.add_scalar(f"Validation_Loss_{target_name}_{kf_count}", val_loss, total_val_step)
        writer.add_scalar(f"Validation_Accuracy_{target_name}_{kf_count}", val_acc/val_count, total_val_step)
        # if model_is_training:
        #     # Save current best model
        #     if val_loss < model.best_val_loss:
        #         model.best_val_loss = val_loss
        #         model.best_epoch = epoch
        #         torch.save(model.state_dict(), f'./best_params_{model.target}.pth')
    return val_acc/val_count, total_val_step


def train(args):
    writer = SummaryWriter("./logs")
    ROOT_DIR = 'D:/Myworks/Learn/Research/EEG'
    RAW_DIR = '/DEAP/data_preprocessed_matlab'
    PROCESSED_DIR = '/Mycode/EEG/ProcessedData/DE'
    # Initialize dataset
    # dataset = DEAPDataset(root=ROOT_DIR,
    #                       raw_dir=RAW_DIR,
    #                       processed_dir=PROCESSED_DIR,
    #                       participant_from=args.participant_from,
    #                       participant_to=args.participant_to)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    # Define loss function
    criterion = torch.nn.BCELoss()
    # Define model targets. Each target has a model associated to it.
    targets = ['valence', 'arousal', 'dominance', 'liking'][:args.n_targets]
    BATCH_SIZE = args.batch_size
    MAX_EPOCH_N = args.max_epoch
    dataset = DEAPDatasetEEGFeatures(root=ROOT_DIR, raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, feature='de')
    # for participant_id in range(1, 33):
    participant_id = 2
    dataset = dataset[40*(participant_id-1):40*participant_id]
    dataset = dataset.shuffle()
    average_accuracy = 0
    target_num = 0
    n_splits = 8
    kf_count = 0
    kf = KFold(n_splits)
    for train_index, val_index in kf.split(dataset):
        print('Now training {} fold'.format(kf_count))
        train_set = dataset.index_select(torch.LongTensor(train_index))
        val_set = dataset.index_select(torch.LongTensor(val_index))
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=args.shuffle_train)
        val_loader = DataLoader(val_set, batch_size=5)

        model = GatedGraphConvGRU(60, 5, 512, 1, 0.5, targets[target_num], training=True).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.000005)
        total_train_step = 0
        total_val_step = 0
        for epoch in range(MAX_EPOCH_N):
            print("------ Epoch {} ------".format(epoch))
            # Train epoch for every model
            total_train_step = train_epoch(kf_count, target_num, targets[target_num], model, train_loader, optim, criterion, device, writer, total_train_step)
            # Validation epoch for every model
            epoch_accuracy, total_val_step = eval_epoch(kf_count, target_num, targets[target_num], model, val_loader, criterion, device, epoch, writer, total_val_step, model_is_training=True)
            average_accuracy = average_accuracy + epoch_accuracy
        kf_count += 1
        # for i, target in enumerate(targets):
        #     print(f'Now training {target} model')
        #     model = GatedGraphConvGRU(60, 5, 512, 1, 0.3, target=target, training=True).to(device)
        #     optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        #     total_train_step = 0
        #     total_val_step = 0
        #     for epoch in range(MAX_EPOCH_N):
        #         print("------ Epoch {} ------".format(epoch))
        #         # Train epoch for every model
        #         total_train_step = train_epoch(i, target, model, train_loader, optim, criterion, device, writer, total_train_step)
        #         # Validation epoch for every model
        #         total_val_step = eval_epoch(i, target, model, val_loader, criterion, device, epoch, writer, total_val_step, model_is_training=True)
    average_accuracy /= MAX_EPOCH_N * n_splits
    print('ACCURACY:{}'.format(average_accuracy))
    writer.close()
