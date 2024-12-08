# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn.init import normal_

class KAF(Module):
    """Basic KAF module. Each activation function is a kernel expansion over a fixed dictionary
    of D elements, and the combination coefficients are learned.
    """

    def __init__(self, num_parameters, D=20, boundary=3.0, init_fcn=None):
        """
        :param num_parameters: number of neurons in the layer.
        :param D: size of the dictionary.
        :param boundary: range of the activation function.
        :param init_fcn: leave to None to initialize randomly, otherwise set a specific function for initialization.
        """
        super(KAF, self).__init__()
        self.num_parameters, self.D = num_parameters, D

        # Initialize the fixed dictionary
        dict_numpy = np.linspace(-boundary, boundary, self.D).astype(np.float32)
        self.register_buffer('dict', torch.from_numpy(dict_numpy))

        # Rule of thumb for gamma
        interval = (self.dict[1] - self.dict[0])
        sigma = 2 * interval  # empirically chosen
        self.gamma = 0.5 / (sigma ** 2)

        # Initialization
        self.init_fcn = init_fcn
        if init_fcn is not None:
            # Initialization with kernel ridge regression
            K = np.exp(-self.gamma * (dict_numpy[:, None] - dict_numpy[None, :]) ** 2)
            self.alpha_init = np.linalg.solve(K + 1e-5, self.init_fcn(dict_numpy)).reshape(-1)
        else:
            self.alpha_init = None

        # Mixing coefficients
        self.alpha = Parameter(torch.empty(1, self.num_parameters, self.D))
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_fcn is not None:
            self.alpha.data = torch.from_numpy(self.alpha_init).repeat(1, self.num_parameters, 1)
        else:
            normal_(self.alpha.data, std=0.3)

    def forward(self, input):
        self.dict = self.dict.to(input.device)  # Ensure dict is on the correct device
        K = torch.exp(-self.gamma * (input.unsqueeze(-1) - self.dict) ** 2)
        y = torch.sum(K * self.alpha, dim=-1)
        return y

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_parameters})"


class KAF2D(Module):
    """Basic 2D-KAF module. Each activation function is a kernel expansion over a pair
    of activation values.
    """

    def __init__(self, num_parameters, D=10, boundary=3.0):
        """
        :param num_parameters: number of neurons (gets halved in output).
        :param D: size of the dictionary.
        :param boundary: range of the activation functions.
        """
        super(KAF2D, self).__init__()
        if num_parameters % 2 != 0:
            raise ValueError('The number of parameters for KAF2D must be even.')

        self.num_parameters = num_parameters // 2
        self.D = D

        # Initialize the fixed dictionary
        x = np.linspace(-boundary, boundary, self.D).astype(np.float32)
        Dx, Dy = np.meshgrid(x, x)
        self.register_buffer('Dx', torch.from_numpy(Dx.ravel()))
        self.register_buffer('Dy', torch.from_numpy(Dy.ravel()))

        # Rule of thumb for gamma
        interval = x[1] - x[0]
        sigma = 2 * interval / np.sqrt(2)  # empirically chosen
        self.gamma = 0.5 / (sigma ** 2)

        # Mixing coefficients
        self.alpha = Parameter(torch.empty(1, self.num_parameters, self.D * self.D))
        self.reset_parameters()

    def reset_parameters(self):
        normal_(self.alpha.data, std=0.3)

    def gauss_2d_kernel(self, X):
        self.Dx = self.Dx.to(X.device)  # Ensure Dx and Dy are on the correct device
        self.Dy = self.Dy.to(X.device)

        X1, X2 = X[:, :self.num_parameters], X[:, self.num_parameters:]
        tmp = (
            -self.gamma * (X1.unsqueeze(-1) - self.Dx) ** 2
            - self.gamma * (X2.unsqueeze(-1) - self.Dy) ** 2
        )
        return torch.exp(tmp)

    def forward(self, input):
        K = self.gauss_2d_kernel(input)
        y = torch.sum(K * self.alpha, dim=-1)
        return y

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_parameters})"
