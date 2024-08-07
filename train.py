#!/usr/bin/env python

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from DEAPDataset2 import DEAPDataset, train_val_test_split, plot_graph, describe_graph, plot_graph
from models.GNNLSTM import GNNLSTM
from matplotlib import pyplot as plt
from tqdm import tqdm


# Define training and eval functions for each epoch (will be shared for all models)
def train_epoch(model, loader, optim, criterion, device):
    # if model.eval_patience_reached:
    # print(2)
    # return -1
    model.train_acc = 0
    model.train()
    epoch_losses = []
    i = 0
    for batch in tqdm(loader):
        # print(batch)
        optim.zero_grad()
        # print(batch.batch)
        batch = batch.to(device)
        # print(batch)
        # out = model(batch, visualize_convolutions=False)
        # print(out)
        # print(out.shape)
        # out=out.squeeze()
        # print(out)
        # print(out.shape)
        # out=out.squeeze()
        # print(out)
        # print(out.shape)
        out = model(batch, visualize_convolutions=False)
        # print(out[0][0])
        # print(out)
        # print(out.shape)
        # Gets first label for every graph
        # print(target)
        target = batch.y.T[model.target].unsqueeze(1)
        # print(target[0][0])
        # print(abs(out[0][0]-target[0][0]))
        j = 0
        k = target.shape[0]
        while j < k:
            i = i + 1
            a = abs(out[j][0] - target[j][0])
            # print(a)
            if a < 4:
                model.train_acc = model.train_acc + 1
            j = j + 1
        # print(target)
        # print(target.shape)
        # print(target.shape[0])
        mse_loss = criterion(out, target)
        # REGULARIZATION
        l1_regularization, l2_regularization = torch.tensor(0, dtype=torch.float).to(device), torch.tensor(0,dtype=torch.float).to(device)
        for param in model.parameters():
            l1_regularization += (torch.norm(param, 1) ** 2).float()
            # l2_regularization += (torch.norm(param, 2)**2).float()
        loss = mse_loss + 0.02 * l1_regularization
        loss.backward()
        optim.step()
        epoch_losses.append(mse_loss.item())
    print(i)
    print(model.train_acc)
    model.train_acc = model.train_acc / i
    epoch_mean_loss = np.array(epoch_losses).mean()
    model.train_losses.append(epoch_mean_loss)
    return epoch_mean_loss


def eval_epoch(model, loader, device, epoch=-1, model_is_training=False, early_stopping_patience=None):
    # if model.eval_patience_reached and model_is_training:
    # return [-1,-1]
    model.eval_acc = 0
    model.eval()
    mses = []
    l1s = []
    # Evaluation
    i = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        target = batch.y.T[model.target].unsqueeze(1)
        j = 0
        k = target.shape[0]
        while j < k:
            i = i + 1
            a = abs(out[j][0] - target[j][0])
            if a < 4:
                model.eval_acc = model.eval_acc + 1
            j = j + 1
        mses.append(F.mse_loss(out, target).item())
        l1s.append(F.l1_loss(out, target).item())
    model.eval_acc = model.eval_acc / i
    print(i)
    e_mse, e_l1 = np.array(mses).mean(), np.array(l1s).mean()
    # Early stopping and checkpoint
    if model_is_training:
        model.eval_losses.append(e_mse)
        # Save current best model locally
        if e_mse < model.best_val_mse:
            model.best_val_mse = e_mse
            model.best_epoch = epoch
            torch.save(model.state_dict(), f'./best_params_{model.target}')
            model.eval_patience_count = 0
        # Early stopping
        elif early_stopping_patience is not None:
            model.eval_patience_count += 1
            if model.eval_patience_count >= early_stopping_patience:
                model.eval_patience_reached = True
    return e_mse, e_l1


def train(args):
    # TODO: set as args
    ROOT_DIR = 'D:/Myworks/Learn/Research/EEG'
    RAW_DIR = '/DEAP/data_preprocessed_matlab'
    PROCESSED_DIR = '/202083270302ywh/GCN-LSTM-deap/ers'
    # Initialize dataset
    dataset = DEAPDataset(root=ROOT_DIR, raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR,
                          participant_from=args.participant_from, participant_to=args.participant_to)
    # print(type(dataset))
    # print(len(dataset))
    # dataset.process()
    # 30 samples are used for training, 5 for validation and 5 are saved for testing
    train_set, val_set, _ = train_val_test_split(dataset)
    # print(train_set)
    # print(val_set)
    # Describe graph structure (same for all instances)
    describe_graph(train_set[0])
    # Set batch size
    BATCH_SIZE = args.batch_size
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=not args.dont_shuffle_train)
    val_loader = DataLoader(val_set, batch_size=1)
    # print(train_loader)
    print(len(train_loader))
    # for batch in val_loader
    # batch
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    # Define loss function
    criterion = torch.nn.MSELoss()

    # Define model targets. Each target has a model associated to it.
    targets = ['valence', 'arousal', 'dominance', 'liking'][:args.n_targets]

    # MODEL PARAMETERS
    # print(train_set[0])
    # print(train_loader[0])
    in_channels = train_set[0].num_node_features
    # print(in_channels)
    # print(in_channels.shape)
    # Print losses over time (train and val)
    plt.figure(figsize=(10, 10))
    # Train models one by one as opposed to having an array [] of models. Avoids CUDA out of memory error
    MAX_EPOCH_N = args.max_epoch
    for i, target in enumerate(targets):
        print(f'Now training {target} model')
        # print(dataset)
        model = GNNLSTM(in_channels, hidden_channels=64, target=target).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        for epoch in range(MAX_EPOCH_N):
            # Train epoch for every model
            t_e_loss = train_epoch(model, train_loader, optim, criterion, device)
            # Validation epoch for every model
            v_e_loss = eval_epoch(model, val_loader, device, epoch, model_is_training=True, early_stopping_patience=4)
            # Break if model has reached patience limit. Model parameters are saved to 'best_params' file.
            if t_e_loss == -1:
                break
            # Epoch results
            print(f'------ Epoch {epoch} ------')
            print(
                f'{target}: Train e. mse: {t_e_loss:.2f} acc:{model.train_acc:.2f}| Validation e. mse: {v_e_loss[0]:.2f} acc:{model.eval_acc:.2f}')
        plt.subplot(2, 2, i + 1)
        plt.plot(model.train_losses)
        plt.plot(model.eval_losses)
        plt.title(f'{target} losses')
        plt.ylabel('loss (mse)')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
    #   # models[i].load_state_dict(torch.load(f'./best_params_{i}'))
    plt.savefig('train_losses.png')

    # Print losses over time (train and val)
    # plt.figure(figsize=(10, 10))
    # Load best performing parameters on validation for each model
    print(f'------ Final model eval ------ \n')
    for i, target in enumerate(targets):
        model = GNNLSTM(in_channels, hidden_channels=64, target=target).to(device)
        model.load_state_dict(torch.load(f'./best_params_{i}'))
        # Evaluating best models
        final_eval = eval_epoch(model, val_loader, device, model_is_training=False)
        print(f'{target} (epoch {model.best_epoch}): Validation mse: {final_eval[0]:.2f} acc:{model.eval_acc:.2f}')
