import numpy as np
import pandas as pd
import random
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset

from models.gat import GATNet, GATFG
from models.gat_gcn import GAT_GCN, GATGCNFG
from models.gcn import GCNNet, GCNFG
from models.ginconv import GINConvNet, GINFG
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.to(device)
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def predicting(model, device, loader):
    model.to(device)
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def set_seed(seed=6):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


datasets = [['kiba', 'bindingdb'][int(sys.argv[1])]]
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet, GINFG, GATFG, GATGCNFG, GCNFG][int(sys.argv[2])]
# modeling = [GCNFG][int(sys.argv[2])]
model_st = modeling.__name__
cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"][int(sys.argv[3])]
print('default cuda_name:', cuda_name)
print('actual cuda name:', cuda_name if torch.cuda.is_available() else 'cpu')

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 300
NUM_CHECKPOINT = 500
KIBA_THRESHOLD = 12.1
BINDINGDB_THRESHOLD = 8
BINDINGDB_THRESHOLD_REVERSE = 16
threshold = [KIBA_THRESHOLD, BINDINGDB_THRESHOLD][int(sys.argv[1])]
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

n_folds = 5
SEED = 6
set_seed(SEED)
metric_root_dir = './metrics/'

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset + '_train')
        test_data = TestbedDataset(root='data', dataset=dataset + '_test')

        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling(dataset=dataset, device=device).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_test_mse = 1000
        best_test_ci = 0
        best_test_aupr = 0
        best_test_r2 = 0
        best_epoch = -1
        best_fold = -1
        model_file_name = 'model_' + model_st + '_' + dataset + '.model'
        result_file_name = 'result_' + model_st + '_' + dataset + '.csv'
        last_metrics_file_name = 'metrics_' + model_st + '_' + dataset + '.csv'
        checkpoint_path = './checkpoints'
        os.makedirs(checkpoint_path, exist_ok=True)

        fold_sizes = [len(train_data) // n_folds] * n_folds
        fold_sizes[-1] += len(train_data) % n_folds  # address data that is indivisible
        folds = torch.utils.data.random_split(train_data, fold_sizes)
        if int(sys.argv[4]) == 0: # normally train and evaluate model
            for i in range(n_folds):
                # valid fold
                valid_data = folds[i]
                # Remaining folds as training set
                train_data = [folds[j] for j in range(5) if j != i]
                train_data = ConcatDataset(train_data)
                # make data PyTorch mini-batch processing ready
                train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
                valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

                for epoch in range(NUM_EPOCHS):
                    train(model, device, train_loader, optimizer, epoch + 1)
                    print('predicting for valid data')
                    G, P = predicting(model, device, valid_loader)
                    val = mse(G, P)
                    if val < best_mse:
                        best_mse = val
                        best_epoch = epoch + 1
                        torch.save(model.state_dict(), model_file_name)
                        print('predicting for test data')
                        G, P = predicting(model, device, test_loader)
                        # ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P), aupr_std(G, P, threshold, dataset), cal_r2_score(G, P)]
                        ret = compute_validation_metrics_parallel(G, P, threshold, dataset)
                        with open(result_file_name, 'w') as f:
                            f.write(','.join(map(str, ret)))
                        best_test_mse = ret[1]
                        best_test_aupr = ret[5][0]
                        best_test_r2 = ret[6][0]
                        best_test_ci = ret[4]
                        print('rmse improved at fold ', i+1, ' epoch ', best_epoch, '; best_test_mse, best_test_ci, aupr, r2: ',
                              best_test_mse,
                              best_test_ci,
                              best_test_aupr,
                              best_test_r2,
                              model_st, dataset)
                    else:
                        print(best_test_mse, 'No improvement since fold ', i+1, ' epoch ', best_epoch, '; best_test_mse, best_test_ci, aupr, r2: ',
                              best_test_mse,
                              best_test_ci,
                              best_test_aupr,
                              best_test_r2,
                              model_st, dataset)
                        # checkpoint
                    if (epoch+1) % NUM_CHECKPOINT == 0:
                        torch.save({'model_dict': model.state_dict(), 'dataset:': dataset, 'model': model_st, 'fold': i, 'epoch': epoch+1}, '{}/checkpoint_{}_{}_{}'.format(checkpoint_path, model_st, dataset, str(epoch+1)))
        elif int(sys.argv[4]) == 1:  # Train and evaluate a classification model
            loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for classification

            for epoch in range(NUM_EPOCHS):
                train(model, device, train_loader, optimizer, epoch + 1)
                print('Predicting for validation data...')
                G, P = predicting(model, device, valid_loader)

                # Convert logits to probabilities and classify
                pred_probs = torch.sigmoid(torch.tensor(P))
                preds = (pred_probs >= 0.5).int().numpy()  # Threshold of 0.5 for binary classification
                labels = G.astype(int)

                # Calculate validation metrics
                acc = accuracy_score(labels, preds)
                precision = precision_score(labels, preds, zero_division=0)
                recall = recall_score(labels, preds, zero_division=0)
                f1 = f1_score(labels, preds, zero_division=0)
                auroc = roc_auc_score(labels, pred_probs.numpy())

                print(
                    f"Epoch {epoch + 1}: Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUROC={auroc:.4f}")

                # Save the best model based on validation AUROC
                if epoch == 0 or auroc > best_test_aupr:
                    best_test_aupr = auroc
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), model_file_name)
                    print(f"Best model saved at epoch {best_epoch} with AUROC={best_test_aupr:.4f}")

            # Load the best model and evaluate on the test set
            print('Evaluating on test data...')
            model.load_state_dict(torch.load(model_file_name, map_location=device))
            G, P = predicting(model, device, test_loader)

            pred_probs = torch.sigmoid(torch.tensor(P))
            preds = (pred_probs >= 0.5).int().numpy()
            labels = G.astype(int)

            # Calculate test metrics
            acc = accuracy_score(labels, preds)
            precision = precision_score(labels, preds, zero_division=0)
            recall = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
            auroc = roc_auc_score(labels, pred_probs.numpy())

            print(
                f"Test Set: Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUROC={auroc:.4f}")

            # Plot AUROC curve
            fpr, tpr, _ = roc_curve(labels, pred_probs.numpy())
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
            plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.savefig(f"./fig/AUROC_{model_st}_{dataset}.png")
            plt.show()

        elif int(sys.argv[4]) < 0:  # merely evaluate model with metrics
            model.load_state_dict(torch.load(model_file_name, map_location=device))
            G, P = predicting(model, device, test_loader)
            aupr, aupr_dev = aupr_std(G, P, threshold, dataset)
            r2, r2_dev = cal_r2_score(G, P)
            rm2 = rm_2_score(G, P)
            plot_auroc(G, P, threshold)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P, device), aupr, aupr_dev, r2, r2_dev, rm2]
            with open(last_metrics_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
        elif int(sys.argv[4]) == 2:
            model.load_state_dict(torch.load(model_file_name, map_location=device))
            G, P = predicting(model, device, test_loader)
            if not os.path.exists(metric_root_dir):
                os.makedirs(metric_root_dir)
            np.savetxt(metric_root_dir + model_st + '_' + dataset.__str__() + '_true', G)
            np.savetxt(metric_root_dir + model_st + '_' + dataset.__str__() + '_pred', P)
            print('Saved metrics to ' + metric_root_dir)
        else:
            assert RuntimeError("Unrecognized argument 4")
