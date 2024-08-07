# -*- coding: utf-8 -*-
# @Time    : 2024/4/24 13:16
# @Author  : HaiqingSun
# @OriginalFileName: FGP
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc
import math

import torch
from torch import nn
import torch.nn.functional as F


def attention(query, key, value, mask, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.w_q = nn.Linear(133, 32)
        self.w_k = nn.Linear(133, 32)
        self.w_v = nn.Linear(133, 32)

        self.dense = nn.Linear(32, 133)
        self.LayerNorm = nn.LayerNorm(133, eps=1e-6)
        self.dropout = nn.Dropout(0.1)

    def forward(self, fg_hiddens, init_hiddens):
        query = self.w_q(fg_hiddens)
        key = self.w_k(fg_hiddens)
        value = self.w_v(fg_hiddens)

        padding_mask = (init_hiddens != 0) + 0.0
        mask = torch.matmul(padding_mask, padding_mask.transpose(-2, -1))
        x, attn = attention(query, key, value, mask)

        hidden_states = self.dense(x)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + fg_hiddens)

        return hidden_states

class Prompt_generator(nn.Module):
    def __init__(self, hidden_size, device):
        super(Prompt_generator, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        # self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.alpha.data.fill_(0.1)
        self.cls = nn.Parameter(torch.randn(1, 133), requires_grad=True)
        self.linear = nn.Linear(133, self.hidden_size)
        self.attention_layer_1 = AttentionLayer(hidden_size)
        self.attention_layer_2 = AttentionLayer(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, fg_states: torch.Tensor, fg_indexs):
        for i in range(len(fg_indexs)):
            fg_states.scatter_(0, fg_indexs[i:i + 1], self.cls)

        hidden_states = self.attention_layer_1(fg_states, fg_states)
        hidden_states = self.attention_layer_2(hidden_states, fg_states)
        # fg_out = torch.zeros(1, self.hidden_size, device=self.device)
        cls_hiddens = torch.gather(hidden_states, 0, fg_indexs)
        cls_hiddens = self.linear(cls_hiddens)
        # fg_hiddens = torch.repeat_interleave(cls_hiddens, torch.tensor(6, device=self.device), dim=0).cuda()
        # fg_out = torch.cat((fg_out, fg_hiddens), 0)


        fg_out = cls_hiddens
        fg_out = self.norm(fg_out)
        return fg_out


class PromptGeneratorOutput(nn.Module):
    def __init__(self, args, self_output):
        super(PromptGeneratorOutput, self).__init__()
        # change position
        self.self_out = self_output
        self.prompt_generator = Prompt_generator(args)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.self_out(hidden_states) # Using a Linear layer(133, 300) for output
        return hidden_states


def prompt_generator_output(args):
    return lambda self_output: PromptGeneratorOutput(args, self_output)


def add_functional_prompt(model, args):
    model.encoder.encoder.W_i_atom = prompt_generator_output(args)(model.encoder.encoder.W_i_atom)
    return model
