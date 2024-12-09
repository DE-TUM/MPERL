#!/usr/bin/env python
# coding: utf-8

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
from torch.autograd import Variable
import numpy as np

class RelationalGraphConvLayer(Module):
    def __init__(self, input_size, output_size, num_bases, num_rel, bias=False, cuda=False):
        super(RelationalGraphConvLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_bases = num_bases
        self.num_rel = num_rel
        self.cuda = cuda
        
        # R-GCN weights
        if num_bases > 0:
            self.w_bases = Parameter(torch.FloatTensor(self.num_bases, self.input_size, self.output_size))
            self.w_rel = Parameter(torch.FloatTensor(self.num_rel, self.num_bases))
        else:
            self.w = Parameter(torch.FloatTensor(self.num_rel, self.input_size, self.output_size))
        # R-GCN bias
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.output_size))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.w_bases.data)
            nn.init.xavier_uniform_(self.w_rel.data)
        else:
            nn.init.xavier_uniform_(self.w.data)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data)
        
    def forward(self, A, X, l):
        X = X.cuda() if X is not None and self.cuda else X
        self.w = torch.einsum('rb, bio -> rio', (self.w_rel, self.w_bases)) if self.num_bases > 0 else self.w
        weights = self.w.view(self.w.shape[0] * self.w.shape[1], self.w.shape[2]) #shape(r*input_size, output_size)
        # Each relations * Weight
        supports = []
        for i in range(self.num_rel):
            if l == 0: # the first layer
                if X is not None:
                    # Yiwen: 2024-09-17 add normalization regarding neighbors
                    tmp_ = torch.sparse.mm(csr2tensor(A[i].multiply(1 / (A[i].sum(axis=1).A1 + 1e-5)[:, None]), self.cuda), X)
                    tmp_ = to_sparse(tmp_)
                else:
                    n_nodes = A[i].shape[0]
                    hidden_dim = self.input_size - n_nodes
                    tmp_ = torch.sparse_coo_tensor(
                        torch.empty((2, 0), dtype=torch.long), 
                        torch.empty(0), 
                        (n_nodes, hidden_dim)) # sparse null_vector h_0
                # Yiwen: 2024-09-20 add gradient detach
                supports.append(torch.cat([csr2tensor(A[i].multiply(1 / (A[i].sum(axis=1).A1 + 1e-5)[:, None]), self.cuda), 
                                           tmp_.detach()], dim=1))
            else:
                # Yiwen: 2024-09-17 add normalization regarding neighbors
                supports.append(torch.sparse.mm(csr2tensor(A[i].multiply(1 / (A[i].sum(axis=1).A1 + 1e-5)[:, None]), self.cuda), X)) # (#node, #node) * (#node, input_size) -> shape(#node, input_size)
        
        # supports (2*#relation, #nodes, input_size)
        tmp = torch.cat(supports, dim=1) # shape(#node, r*input_size)
        out = torch.mm(tmp.float(), weights) #(#node, r*input_size) * (r*input_size, output_size) -> shape(#node, output_size)

        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out

class RelGraphConvLayer(Module):
    def __init__(
        self, input_size, output_size, num_bases, num_rel, bias=False, cuda=False
    ):
        super(RelGraphConvLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_bases = num_bases
        self.num_rel = num_rel
        self.cuda = cuda

        # R-GCN weights
        if num_bases > 0:
            self.w_bases = Parameter(
                torch.FloatTensor(self.num_bases, self.input_size, self.output_size)
            )
            self.w_rel = Parameter(torch.FloatTensor(self.num_rel, self.num_bases))
        else:
            self.w = Parameter(
                torch.FloatTensor(self.num_rel, self.input_size, self.output_size)
            )
        # R-GCN bias
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.output_size))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.w_bases.data)
            nn.init.xavier_uniform_(self.w_rel.data)
        else:
            nn.init.xavier_uniform_(self.w.data)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data)

    def forward(self, A, X):
        X = X.cuda() if X is not None and self.cuda else X
        self.w = (
            torch.einsum("rb, bio -> rio", (self.w_rel, self.w_bases))
            if self.num_bases > 0
            else self.w
        )
        weights = self.w.view(
            self.w.shape[0] * self.w.shape[1], self.w.shape[2]
        )  # shape(r*input_size, output_size)
        # Each relations * Weight
        supports = []
        for i in range(self.num_rel):
            if X is not None:
                supports.append(torch.sparse.mm(csr2tensor(A[i], self.cuda), X))
            else:
                supports.append(csr2tensor(A[i], self.cuda))

        tmp = torch.cat(supports, dim=1)
        out = torch.mm(tmp.float(), weights)  # shape(#node, output_size)

        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out

def to_sparse(x):
        """ converts dense tensor x to sparse format """
        x_typename = torch.typename(x).split('.')[-1]
        sparse_tensortype = getattr(torch.sparse, x_typename)

        indices = torch.nonzero(x)
        if len(indices.shape) == 0:  # if all elements are zeros
            return sparse_tensortype(*x.shape)
        indices = indices.t()
        values = x[tuple(indices[i] for i in range(indices.shape[0]))]
        return sparse_tensortype(indices, values, x.size())     

def csr2tensor(A, cuda):
        coo = A.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        if cuda:
            out = torch.sparse.FloatTensor(i, v, torch.Size(shape)).cuda()
        else:
            out = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        return out


