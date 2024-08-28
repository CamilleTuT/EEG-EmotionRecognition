import torch
import numpy as np
from torch_geometric.loader import DataLoader
from DEAPDataset_Spacial import DEAPDataset, train_val_test_split
from DEAPDataset_freq import DEAPDatasetEEGFeatures
from models.GNNLSTM import GNNLSTM
from models.GatedGraphConvGRU import GatedGraphConvGRU
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# Define training and eval functions for each epoch (will be shared for all models)
def train_epoch(targetname, model, loader, optim, criterion, device, writer, total_train_step):
    model.train()
    model.training = True
    epoch_losses = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        target = batch.y.T[model.target].unsqueeze(1)
        output = model(batch)
        mse_loss = criterion(output, target)
        # REGULARIZATION
        l1_regularization = torch.tensor(0, dtype=torch.float).to(device)
        # l2_regularization = torch.tensor(0, dtype=torch.float).to(device)
        for param in model.parameters():
            l1_regularization += (torch.norm(param, 1) ** 2).float()
            # l2_regularization += (torch.norm(param, 2)**2).float()
        loss = mse_loss + 0.02 * l1_regularization
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step += 1
        if total_train_step % 64 == 0:
            print("Train Batch Counter: {}, Loss: {}".format(total_train_step, mse_loss.item()))
            writer.add_scalar(f"Train Loss_{targetname}", mse_loss.item(), total_train_step)

        epoch_losses.append(mse_loss.item())
    epoch_mean_loss = np.array(epoch_losses).mean()
    model.train_losses.append(epoch_mean_loss)
    return total_train_step, epoch_mean_loss


def eval_epoch(targetname, model, loader, criterion, device, epoch, writer, total_val_step, model_is_training):
    model.eval()
    model.training = False
    val_count = 0
    val_loss = 0
    val_acc = 0
    # Evaluation
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            print(output)
            target = batch.y.T[model.target].unsqueeze(1)
            batch_size = target.shape[0]
            batch_count = 0
            loss = criterion(output, target)
            val_loss += loss.item()
            total_val_step += 1
            while batch_count < batch_size:
                if output[batch_count][0] < 5.5 and target[batch_count][0] < 5.5:
                    val_acc += 1
                elif output[batch_count][0] > 5.5 and target[batch_count][0] > 5.5:
                    val_acc += 1
                batch_count += 1
                val_count += 1

        print("Loss on the validation set: {}".format(val_loss))
        print("Accuracy on the validation set: {}".format(val_acc/val_count))
        writer.add_scalar(f"Validation_Loss_{targetname}", val_loss, total_val_step)
        writer.add_scalar(f"Validation_Accuracy_{targetname}", val_acc/val_count, total_val_step)

        if model_is_training:
            # Save current best model
            if val_loss < model.best_val_loss:
                model.best_val_loss = val_loss
                model.best_epoch = epoch
                torch.save(model.state_dict(), f'./best_params_{model.target}.pth')
    return total_val_step

def train(args):
    writer = SummaryWriter("./logs")
    ROOT_DIR = 'D:/Myworks/Learn/Research/EEG'
    RAW_DIR = '/DEAP/data_preprocessed_matlab'
    PROCESSED_DIR = '/202083270302ywh/GCN-LSTM-deap/ProcessedData/Wavelet'
    # Initialize dataset
    # dataset = DEAPDataset(root=ROOT_DIR,
    #                       raw_dir=RAW_DIR,
    #                       processed_dir=PROCESSED_DIR,
    #                       participant_from=args.participant_from,
    #                       participant_to=args.participant_to)
    dataset = DEAPDatasetEEGFeatures(root=ROOT_DIR, raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, feature='wav')
    # print(dataset[0].x.shape) torch.Size([32, 7680])
    # print(dataset[0].y.shape) torch.Size([1, 4])
    # print(dataset.x.shape) torch.Size([40960, 7680])
    # print(dataset.y.shape) torch.Size([1280, 4])
    # 30 for training 5 for validation 5 for testing
    train_set, val_set, _ = train_val_test_split(dataset)
    BATCH_SIZE = args.batch_size
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=not args.dont_shuffle_train)
    val_loader = DataLoader(val_set, batch_size=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    # Define loss function
    criterion = torch.nn.MSELoss()
    # Define model targets. Each target has a model associated to it.
    targets = ['valence', 'arousal', 'dominance', 'liking'][:args.n_targets]
    # Train models one by one as opposed to having an array [] of models. Avoids CUDA out of memory error
    MAX_EPOCH_N = args.max_epoch
    for i, target in enumerate(targets):
        print(f'Now training {target} model')
        # model = GNNLSTM(target=target, training=True).to(device)
        model = GatedGraphConvGRU(59, 4, 128, 2, 0.4, target=target, training=True).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        total_train_step = 0
        total_val_step = 0
        for epoch in range(MAX_EPOCH_N):
            print("------ Epoch {} ------".format(epoch))

            # Train epoch for every model
            total_train_step, t_e_loss = train_epoch(target, model, train_loader, optim, criterion, device, writer, total_train_step)

            # Validation epoch for every model
            total_val_step = eval_epoch(target, model, val_loader, criterion, device, epoch, writer, total_val_step, model_is_training=True)

            if t_e_loss == -1:
                break

    writer.close()
