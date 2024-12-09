#!/usr/bin/env python
# coding: utf-8

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import scipy.sparse as sp
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
from torch.autograd import Variable
from src.layers import *




class PonderRelationalGraphConvModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_bases, num_rel, num_layer, dropout, max_steps=1, featureless=True, cuda=False, seed=0):
        """
        * `n_elems` is the number of elements in the input vector --> input_size
        * `n_hidden` is the state vector size of the feature representation --> hidden_size
        * `max_steps` is the maximum number of steps $N$
        """
        #super().__init__()
        torch.manual_seed(seed)
        super(PonderRelationalGraphConvModel, self).__init__()
        
        self.hidden_size = hidden_size

        self.lambda_layer = nn.Linear(output_size, 1)
        self.lambda_prob = nn.Sigmoid()
        # An option to set during inference so that computation is actually halted at inference time
        

        self.input_size = input_size
        self.output_size = output_size
        self.max_steps = max_steps
        self.num_layer = num_layer
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        self.is_halt = False
        #self.softmax = nn.Softmax()

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        for i in range(self.num_layer):
            if i == 0:
                # Yiwen: change input layer size
                self.layers.append(RelationalGraphConvLayer(input_size+output_size, hidden_size,
                                                                num_bases, num_rel, bias=False, cuda=cuda))
                # self.layers.append(RelationalGraphConvLayer(input_size, hidden_size, 
                #                                             num_bases, num_rel, bias=False, cuda=cuda))
            else:
                if i == self.num_layer-1:
                    self.layers.append(RelationalGraphConvLayer(hidden_size, output_size,
                                                                num_bases, num_rel, bias=False, cuda=cuda))
                else:
                    self.layers.append(RelationalGraphConvLayer(hidden_size, hidden_size, 
                                                                num_bases, num_rel, bias=False, cuda=cuda))



    def forward(self, A, X):
        """
        * `x` is the input of shape `[batch_size, n_elems]`

        This outputs a tuple of four tensors:

        1. $p_1 \dots p_N$ in a tensor of shape `[N, batch_size]`
        2. $\hat{y}_1 \dots \hat{y}_N$ in a tensor of shape `[N, batch_size]` - the log probabilities of the parity being $1$
        3. $p_m$ of shape `[batch_size]`
        4. $\hat{y}_m$ of shape `[batch_size]` where the computation was halted at step $m$
        """

        batch_size = self.input_size
        
        # h = None # featureless -> null vector
        h = torch.ones(batch_size, self.output_size)
        for i, layer in enumerate(self.layers):
            h = layer(A, h, i)
            if i != self.num_layer-1:
                h = F.dropout(self.relu(h), self.dropout, training=self.training)
            else:
                h = F.dropout(h, self.dropout, training=self.training)

        # initialize the probability of halting at step 0
        p = []
        y = []
        lamda = []
        # $\prod_{j=1}^{n-1} (1 - \lambda_j)$
        un_halted_prob = h.new_ones((batch_size,))

        # A vector to maintain which samples has halted computation
        halted = h.new_zeros((batch_size,))
        p_m = h.new_zeros((batch_size,))
        y_m = h.new_zeros((batch_size,))

        # Iterate for $N$ steps
        cur_max_steps = self.max_steps
        # cur_max_steps = self.max_steps if self.training else 1
        for n in range(1, cur_max_steps + 1):

            # The halting probability $\lambda_N = 1$ for the last step
            if n == cur_max_steps:
                lambda_n = h.new_ones(h.shape[0])
            else:
                lambda_n = self.lambda_prob(self.lambda_layer(h))[:, 0]

            y_n = h[:, 0]

            # $$p_n = \lambda_n \prod_{j=1}^{n-1} (1 - \lambda_j)$$
            p_n = un_halted_prob * lambda_n
            # Update $\prod_{j=1}^{n-1} (1 - \lambda_j)$
            un_halted_prob = un_halted_prob * (1 - lambda_n)

            # Halt based on halting probability $\lambda_n$
            # print("Current lambda_n value: ", lambda_n, '\t', lambda_n.size())
            halt = torch.bernoulli(lambda_n) * (1 - halted)

            # Yiwen: Collect $p_n$ and $\hat{y}_n$ and also $\lambda_n$
            p.append(p_n)
            y.append(h)
            lamda.append(lambda_n)

            # Update $p_m$ and $\hat{y}_m$ based on what was halted at current step $n$
            p_m = p_m * (1 - halt) + p_n * halt
            y_m = y_m * (1 - halt) + y_n * halt

            # Update halted samples
            halted = halted + halt

            # Stop the computation if all samples have halted
            if halted.sum() == batch_size:
                break

            # Yiwen: Get next state $h_{n+1} = s_h(x, h_n)$
            # h = torch.cat((x_one_hot, h), 1) # do concatenation within the layer
            for i, layer in enumerate(self.layers):
                h = layer(A, h, i)
                if i != self.num_layer-1:
                    h = F.dropout(self.relu(h), self.dropout, training=self.training)
                else:
                    h = F.dropout(h, self.dropout, training=self.training)


        return torch.stack(y), torch.stack(p), torch.stack(lamda)
    
class RelationalGraphConvModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_bases,
        num_rel,
        num_layer,
        dropout,
        featureless=True,
        cuda=False,
    ):
        super(RelationalGraphConvModel, self).__init__()

        self.num_layer = num_layer
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()

        for i in range(self.num_layer):
            if i == 0:
                self.layers.append(
                    RelGraphConvLayer(
                        input_size,
                        hidden_size,
                        num_bases,
                        num_rel,
                        bias=False,
                        cuda=cuda,
                    )
                )
            else:
                if i == self.num_layer - 1:
                    self.layers.append(
                        RelGraphConvLayer(
                            hidden_size,
                            output_size,
                            num_bases,
                            num_rel,
                            bias=False,
                            cuda=cuda,
                        )
                    )
                else:
                    self.layers.append(
                        RelGraphConvLayer(
                            hidden_size,
                            hidden_size,
                            num_bases,
                            num_rel,
                            bias=False,
                            cuda=cuda,
                        )
                    )

    def forward(self, A, X):
        # x = X
        x = None  # featureless
        for i, layer in enumerate(self.layers):
            x = layer(A, x)
            if i != self.num_layer - 1:
                x = F.dropout(self.relu(x), self.dropout, training=self.training)
            else:
                x = F.dropout(x, self.dropout, training=self.training)
        return x
    

class ReconstructionLoss(Module):
    """
    ## Reconstruction loss
    $$L_{Rec} = \sum_{n=1}^N p_n \mathcal{L}(y, \hat{y}_n)$$
    $\mathcal{L}$ is the normal loss function between target $y$ and prediction $\hat{y}_n$.
    """

    def __init__(self, loss_func: nn.Module):
        """
        * `loss_func` is the loss function $\mathcal{L}$
        """
        super().__init__()
        self.loss_func = loss_func

    def forward(self, p: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        """
        * `p` is $p_1 \dots p_N$ in a tensor of shape `[N, batch_size]`
        * `y_hat` is $\hat{y}_1 \dots \hat{y}_N$ in a tensor of shape `[N, batch_size, ...]`
        * `y` is the target of shape `[batch_size, ...]`
        """

        # The total $\sum_{n=1}^N p_n \mathcal{L}(y, \hat{y}_n)$
        total_loss = p.new_tensor(0.)
        # Iterate upto $N$
        for n in range(p.shape[0]):
            # $p_n \mathcal{L}(y, \hat{y}_n)$ for each sample and the mean of them
            loss = (p[n] * self.loss_func(y_hat[n], y)).mean()
            # Add to total loss
            total_loss = total_loss + loss

        #
        return total_loss


class RegularizationLoss(Module):
    """
    ## Regularization loss
    $$L_{Reg} = \mathop{KL} \Big(p_n \Vert p_G(\lambda_p) \Big)$$
    $\mathop{KL}$ is the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).
    $p_G$ is the [Geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution) parameterized by
    $\lambda_p$. *$\lambda_p$ has nothing to do with $\lambda_n$; we are just sticking to same notation as the paper*.
    $$Pr_{p_G(\lambda_p)}(X = k) = (1 - \lambda_p)^k \lambda_p$$.
    The regularization loss biases the network towards taking $\frac{1}{\lambda_p}$ steps and incentivies non-zero probabilities
    for all steps; i.e. promotes exploration.
    """

    def __init__(self, lambda_p: float, max_steps: int = 1):
        """
        * `lambda_p` is $\lambda_p$ - the success probability of geometric distribution
        * `max_steps` is the highest $N$; we use this to pre-compute $p_G(\lambda_p)$
        """
        super().__init__()

        # Empty vector to calculate $p_G(\lambda_p)$
        p_g = torch.zeros((max_steps,))
        # $(1 - \lambda_p)^k$
        not_halted = 1.
        # Iterate upto `max_steps`
        for k in range(max_steps):
            # $$Pr_{p_G(\lambda_p)}(X = k) = (1 - \lambda_p)^k \lambda_p$$
            p_g[k] = not_halted * lambda_p
            # Update $(1 - \lambda_p)^k$
            not_halted = not_halted * (1 - lambda_p)

        # Save $Pr_{p_G(\lambda_p)}$
        self.p_g = nn.Parameter(p_g, requires_grad=False)

        # KL-divergence loss
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p: torch.Tensor, cuda=False):
        """
        * `p` is $p_1 \dots p_N$ in a tensor of shape `[N, batch_size]`
        """
        # Transpose `p` to `[batch_size, N]`
        p = p.transpose(0, 1)
        # Get $Pr_{p_G(\lambda_p)}$ upto $N$ and expand it across the batch dimension
        p_g = self.p_g[None, :p.shape[1]].expand_as(p)
        if cuda:
            p_g = p_g.cuda()
            p = p.cuda()
        # Calculate the KL-divergence.
        # *The [PyTorch KL-divergence](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
        # implementation accepts log probabilities.*
        epsilon = 1e-8 # Yiwen: avoid log(0)
        return self.kl_div((p + epsilon).log(), p_g)
