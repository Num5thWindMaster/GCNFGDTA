import os
import pickle

import numpy as np
from math import sqrt

import sklearn
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Mol, MolFromSmiles
from scipy import stats
from sklearn.metrics import roc_auc_score, r2_score, precision_recall_curve, average_precision_score, roc_curve, auc as ori_auc
from torch import nn, Tensor
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch


class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, pred=False):

        # root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]) and not pred:
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif os.path.isfile(self.processed_paths[0]) and pred:
            print('Rebuild pre-processed data, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif not os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]),
                                smiles=smiles)
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])


def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def auc(y, f):
    return roc_auc_score(y, f)


def cal_r2_score(y, f):
    r2 = r2_score(y, f)
    n_iterations = 100  # 进行100次重复计算
    r2_values = np.zeros(n_iterations)

    for i in range(n_iterations):
        # 生成随机的预测结果
        random_pred = np.random.rand(len(y))
        # 计算 R2
        r2_values[i] = r2_score(y, random_pred)

    r2_std = r2, np.std(r2_values)
    return r2_std


def r0_2_score(y_true, y_pred):
    y_true_centered = y_true - np.mean(y_true)
    y_pred_centered = y_pred - np.mean(y_pred)
    r0_2 = r2_score(y_true_centered, y_pred_centered)
    return r0_2


def rm_2_score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    r0_2 = r0_2_score(y_true, y_pred)
    return r2 * (1 - np.sqrt(np.abs(r2 - r0_2)))


def aupr_std(y, f, threshold):
    y_bin_true = np.where(y >= threshold, 1, 0)
    ap = average_precision_score(y_bin_true, f)
    # 重复计算多个预测结果的平均精确度和标准差
    n_iterations = 100  # 进行100次重复计算
    ap_values = np.zeros(n_iterations)

    for i in range(n_iterations):
        # 生成随机的预测结果
        random_pred = np.random.rand(len(y_bin_true))
        # 计算平均精确度
        ap_values[i] = average_precision_score(y_bin_true, random_pred)

    return ap, np.std(ap_values)

def plot_auroc(y, f, threshold):
    y_bin_true = np.where(y >= threshold, 1, 0)
    fpr, tpr, thresholds = roc_curve(y_bin_true, f)
    roc_auc = ori_auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def load_embedding_from_pkl(path):
    # 读取包含键值对的文件
    # embedding_dict = {}
    # with open(path, "r") as file:
    #     for line in file:
    #         key, *value = line.split()  # 假设每行的格式为 "key value1 value2 ..."
    #         embedding_dict[key] = np.array([float(val) for val in value])
    embedding_dict = pickle.load(open(path, 'rb'))
    # 从embedding_dict中获取词汇表大小和嵌入维度
    num_embeddings = len(embedding_dict)

    # embedding_dim = len(embedding_dict.values())
    embedding_dim = len(next(iter(embedding_dict.values())))

    # 创建嵌入层对象，并加载权重
    embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
    for idx, (key, value) in enumerate(embedding_dict.items()):
        embedding_layer.weight.data[idx] = torch.tensor(value).to('cpu')

    # 冻结权重
    embedding_layer.weight.requires_grad = False
    return embedding_layer
    # 示例：使用嵌入层将索引为0的词汇转换为嵌入向量
    # index_tensor = torch.tensor([0])  # 假设将索引为0的词汇转换为嵌入向量
    # embedded_vector = embedding_layer(index_tensor)
    #
    # print(embedded_vector)


def match_fg(mol: Mol):
    fg2emb = pickle.load(open('initial/fg2emb.pkl', 'rb'))
    with open('initial/funcgroup.txt', "r") as f:
        funcgroups = f.read().strip().split('\n')
        name = [i.split()[0] for i in funcgroups]
        smart = [Chem.MolFromSmarts(i.split()[1]) for i in funcgroups]
        smart2name = dict(zip(smart, name))
    fg_emb = [[1] * 133]
    pad_fg = [[0] * 133]
    for sm in smart:
        if mol.HasSubstructMatch(sm):
            fg_emb.append(fg2emb[smart2name[sm]].tolist())
    if len(fg_emb) > 13:
        fg_emb = fg_emb[:13]
    else:
        fg_emb.extend(pad_fg * (13 - len(fg_emb)))
    return fg_emb


def calculate_fg(self, data, device):
    smiles = data.smiles
    embs = []
    for smile in smiles:
        mol = MolFromSmiles(smile)
        emb = match_fg(mol)
        embs.extend(emb)
    embs = Tensor(embs)
    embs = embs.to(device)

    fg_index = [i * 13 for i in range(len(smiles))]
    fg_indxs = [[i] * 133 for i in fg_index]
    fg_indxs = torch.LongTensor(fg_indxs)
    fg_indxs = fg_indxs.to(device)
    # if torch.cuda.is_available():
    #     fg_indxs = fg_indxs.cuda(device)
    #     embs = embs.cuda(device)

    output_fg = self.prompt(embs, fg_indxs)
    output_fg = self.fg_out_linear(output_fg)
    return output_fg
