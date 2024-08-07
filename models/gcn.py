import pickle

import torch
import torch.nn as nn
from rdkit.Chem import MolFromSmiles
from torch import Tensor
from torch_geometric.nn import GCNConv, global_max_pool as gmp

from models.FGP import Prompt_generator
from utils import load_embedding_from_pkl, match_fg, calculate_fg


# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=25, output_dim=128,
                 dropout=0.2):
        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32 * 121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        target = data.target

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)  # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


class GCNFG(GCNNet):
    def __init__(self, fg_input_size=-1, fg_embed_size=82, fg_num_heads=1, fg_hidden_dim=100,
                 input_indices=torch.arange(82), fg_hidden_size=300, fg_output_size=100, n_output=1,
                 n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=25, output_dim=128,
                 dropout=0.2, device=torch.device('cpu')):
        super(GCNFG, self).__init__(n_output, n_filters, embed_dim, num_features_xd, num_features_xt,
                                    output_dim,
                                    dropout)
        self.device = device

        # GCNFG
        self.prompt = Prompt_generator(fg_hidden_size, device)
        self.fg_out_linear = nn.Linear(300, output_dim, bias=True)

        self.fc1 = nn.Linear(3 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch, smiles = data.x, data.edge_index, data.batch, data.smiles
        # get protein input
        target = data.target

        # GCNNet branch
        x_gcn = self.conv1(x, edge_index)
        x_gcn = self.relu(x_gcn)

        x_gcn = self.conv2(x_gcn, edge_index)
        x_gcn = self.relu(x_gcn)

        x_gcn = self.conv3(x_gcn, edge_index)
        x_gcn = self.relu(x_gcn)
        x_gcn = gmp(x_gcn, batch)  # global max pooling

        x_gcn = self.relu(self.fc_g1(x_gcn))
        x_gcn = self.dropout(x_gcn)
        x_gcn = self.fc_g2(x_gcn)
        x_gcn = self.dropout(x_gcn)

        # Protein sequence branch
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # GCNFG branch
        output_fg = calculate_fg(self, data, device=self.device)

        # Concatenate outputs
        xc = torch.cat((x_gcn, xt, output_fg), 1)

        # Dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


