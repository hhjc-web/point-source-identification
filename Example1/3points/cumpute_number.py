import numpy as np
import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
import time
from pyDOE import lhs
import matplotlib.pyplot as plt
import matplotlib.ticker
import math

def compute_number(X_u_train, u_train, un_train, N, M):
    dim = 2
    boundary_size = dim * 2**dim
    alpha = np.zeros(2*M-1, dtype = 'complex_')
    for i in range(M):
        for k in range(4):
            x = X_u_train[k*N:(k+1)*N,:]
            alpha[i] = alpha[i] + trapz(-un_train[k*N:(k+1)*N,:] * z(x, i) + u_train[k*N:(k+1)*N,:] * z_n(x, i))
    mu = np.zeros((M,1), dtype = 'complex_')
    for k in range(M):
        mu[k,0] = alpha[k]
    for i in range(M,2*M-1):
        for k in range(4):
            x = X_u_train[k*N:(k+1)*N,:]
            alpha[i] = alpha[i] + trapz(-un_train[k*N:(k+1)*N,:] * z(x, i) + u_train[k*N:(k+1)*N,:] * z_n(x, i))
        mu_temp = np.zeros((M, 1), dtype='complex_')
        for k in range(M):
            mu_temp[k, 0] = alpha[k+i+1-M]
        mu = np.hstack((mu, mu_temp))
        _, singularv, _ = np.linalg.svd(mu)
        print(singularv)
        if np.min(singularv) < 1.5:
            break
    rank = i+1-M
    return rank

def z(X_u_train, i):
    S = (X_u_train[:,0] + X_u_train[:,1] * 1j)**i
    S = torch.reshape(S, (S.shape[0], 1))
    return S

def z_n(X_u_train, i):
    grad_x = i * (X_u_train[:, 0] + X_u_train[:, 1] * 1j) ** (i - 1)
    grad_x = torch.reshape(grad_x, (grad_x.shape[0], 1))
    grad_y = 1j * i * (X_u_train[:, 0] + X_u_train[:, 1] * 1j) ** (i - 1)
    grad_y = torch.reshape(grad_y, (grad_y.shape[0], 1))
    grad = torch.hstack((grad_x,grad_y))
    outernormal = torch.ceil((X_u_train + 1) / 2) + torch.floor((X_u_train + 1) / 2) - 1
    outernormal[0] = outernormal[1]
    outernormal[-1] = outernormal[1]
    result = torch.sum(grad*outernormal, dim = 1, keepdim = True)
    return result

def trapz(u):
    return 2/(u.shape[0]-1) * (torch.sum(u)-0.5*u[0]-0.5*u[-1])