import numpy as np
import torch
import sys

from torch_geometric.data import DataLoader

from create_data import smile_to_graph, seq_cat
from models.gat import GATNet, GATFG
from models.gat_gcn import GAT_GCN, GATGCNFG
from models.gcn import GCNNet, GCNFG
from models.ginconv import GINConvNet, GINFG
from utils import TestbedDataset

dataset = ['kiba', 'bindingdb'][int(sys.argv[1])]
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet, GINFG, GATFG, GATGCNFG, GCNFG][int(sys.argv[2])]
model_st = modeling.__name__
model_file_name = 'model_' + model_st + '_' + dataset + '.model'
input_file = 'predict_data/test.txt'
TEST_BATCH_SIZE = 512
device = torch.device('cpu')

model = modeling(device=device)
model.load_state_dict(torch.load(model_file_name, map_location=device))
model.to(device)
model.eval()

xd = []
xt = []
y = []
smile_graph = {}
with open(input_file, 'r', encoding='utf-8') as f:
    for pair in f.readlines():
        if pair:
            pair = pair.strip().split()
            xd.append(pair[0])
            xt.append(pair[1])
            y.append(-1)
            smile_graph[pair[0]] = smile_to_graph(pair[0])

xd = np.asarray(xd)
XT = [seq_cat(t) for t in xt]
xt = np.asarray(XT)
y = np.asarray(y)

KIBA_THRESHOLD = 12.1
BINDINGDB_THRESHOLD = 8.0
threshold = [KIBA_THRESHOLD, BINDINGDB_THRESHOLD][int(sys.argv[1])]

test_data = TestbedDataset(root='test', dataset='kiba', xd=xd, xt=xt, y=y, smile_graph=smile_graph, pred=True)
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
total_preds = torch.Tensor()
total_labels = torch.Tensor()
print('Make prediction for {} samples...'.format(len(test_loader.dataset)))
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        total_preds = torch.cat((total_preds, output.cpu()), 0)
        total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
        output = total_preds.numpy().flatten()
        count = np.count_nonzero(total_preds >= threshold)
        indices = np.where(total_preds >= threshold)[0]
print(len(output))
print(count)
print(indices)
print(output)
indices = indices.tolist()
output = output.tolist()
print(len(indices))
print(len(output))
filename = "result_" + str(threshold) + "_" + model_st + ".csv"
with open(filename, 'w') as file:
    file.write(','.join(map(str, indices)))
    file.write('\n')
    file.write(','.join(map(str, output)))

