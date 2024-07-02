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
from math import floor
import os
import scipy.io as sio
from typing import Union, Dict, Optional, Tuple, List
from typing import Callable
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Set default dtype to float32
torch.set_default_dtype(torch.float)
#PyTorch random number generator
torch.manual_seed(1234)
# Random number generators in other libraries
np.random.seed(1234)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device == 'cuda': print(torch.cuda.get_device_name())

output_path = '.'
log_path = os.path.join(output_path, "log.txt")
loss_path = os.path.join(output_path, "loss.mat")

def laplace(func: Callable[[torch.Tensor], torch.Tensor], x):
    xclone = x.clone().detach().requires_grad_(True)
    uforward = func(xclone)

    grad = autograd.grad(uforward, xclone, grad_outputs=torch.ones_like(uforward),\
        only_inputs=True, create_graph=True, retain_graph=True)[0]

    result = autograd.grad(grad[:, 0:1], xclone, grad_outputs=torch.ones_like(grad[:, 0:1]),\
        only_inputs=True, create_graph=True, retain_graph=True)[0][:, 0:1]
    for k in range(1, x.size(1)):
        result = torch.cat([result, autograd.grad(grad[:, k:(k+1)], xclone, grad_outputs=torch.ones_like(grad[:, k:(k+1)]),\
        only_inputs=True, create_graph=True, retain_graph=True)[0][:, k:(k+1)]], dim = 1)

    result = torch.sum(result, dim=1, keepdim = True)
    return result

# Data Prep
def _generate_uniform_mesh(numEachDim: int, dim1: int, dim2: int, totalDim: int) -> Tensor:
    grid_tmp = torch.linspace(-1.0, 1.0, numEachDim + 1)
    grid_x1, grid_x2 = torch.meshgrid([grid_tmp, grid_tmp], indexing='xy')
    grid_x1, grid_x2 = grid_x1.reshape((-1, 1)), grid_x2.reshape((-1, 1))
    result = torch.ones((numEachDim + 1)*(numEachDim + 1), totalDim) * 0.0  # slice at x3=x4=...=0
    result[:, dim1:(dim1 + 1)] = grid_x1
    result[:, dim2:(dim2 + 1)] = grid_x2
    return result

# Test Data

ndim = 10
X_u_test = _generate_uniform_mesh(200, 0, 1, ndim)
X_u_test = X_u_test.to(device)
lb = -1 * np.ones(ndim)  # lower bound
ub = np.ones(ndim)  # upper bound

# source term

def solution_v(x: torch.Tensor) -> torch.Tensor:
    result = torch.sum(x**2, dim = 1, keepdim = True)
    return result - ndim * x[:, 0:1]**2

c = np.matrix([100,100])
xs = np.matrix([[-1/2,-1/2,0,0,0,0,0,0,0,0],[1/2,1/2,0,0,0,0,0,0,0,0]])
c = torch.from_numpy(c).float().to(device)
xs = torch.from_numpy(xs).float().to(device)

n = 2
def solution(x: torch.Tensor) -> torch.Tensor:
    usol = solution_v(x)
    for i in range(n):
        xs_temp = torch.matmul(torch.ones([x.shape[0], 1]).to(device), xs[i:i+1,:])
        r = torch.sum((x - xs_temp)**2, dim = 1, keepdim =True)
        PHI = c[0, i] * 3 /(2 * math.pi**5 * r**4)
        usol = usol + PHI
    return usol

vsol = solution_v(X_u_test)
usol = solution(X_u_test)

# Training Data

def trainingdata(N_f, N_v):
    '''Collocation Points'''
    # N_f sets of tuples(x,t)
    X_f = lb + (ub - lb) * lhs(ndim, N_f)

    X_u_train = lb + (ub - lb) * lhs(ndim, N_v)
    for index in range(N_v):
        dimIndex = floor(ndim * index / N_v)
        X_u_train[index, dimIndex] = (index % 2) * 2 - 1

    X_f_train = np.vstack((X_f, X_u_train))  # append training points to collocation points

    X_f_train = torch.from_numpy(X_f_train).float().to(device)
    X_u_train = torch.from_numpy(X_u_train).float().to(device)
    u_train = solution(X_u_train)

    outernormal = torch.ceil((X_u_train+1)/2) + torch.floor((X_u_train+1)/2) - 1.0
    g = X_u_train.clone()
    g.requires_grad = True
    un_train = torch.sum(autograd.grad(solution(g), g, torch.ones([g.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0] * outernormal, dim =1, keepdim=True)

    return X_f_train, X_u_train, u_train, un_train

# SSPINN

class Sequentialmodel(nn.Module):

    def __init__(self, layers):
        super().__init__()  # call __init__ from parent class
        'activation function'
        self.activation = nn.Tanh()
        'loss function'
        self.loss_function = nn.MSELoss(reduction='mean')
        'Initialise neural network as a nn.MSELosslist using nn.Modulelist'
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        # self.x = [0,0,0,0]
        # self.c = [0,0,0,0]
        self.n_pred = n
        with open(log_path, "w") as f:
            f.write("Number of sources:{:d} \n" .format(self.n_pred))
        rand_point = 2*torch.rand(self.n_pred, ndim) - 1
        self.x = torch.autograd.Variable(rand_point.to(device), requires_grad=True)
        self.c = torch.autograd.Variable(torch.rand(self.n_pred, 1).to(device), requires_grad=True)
        'Xavier Normal Initialization'
        # std = gain * sqrt(2/(input_dim+output_dim))
        for i in range(len(layers) - 1):
            # weights from a normal distribution with
            # Recommended gain value for tanh = 5/3?
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)
        # preprocessing input
        x = (x - l_b) / (u_b - l_b)  # feature scaling
        # convert to float
        a = x.float()
        for i in range(len(layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    def loss_BC(self, x, y):
        for i in range(self.n_pred):
            xs_temp = torch.matmul(torch.ones([N_v, 1]).to(device), self.x[i:i+1])
            r = torch.sum((x - xs_temp)**2, dim=1, keepdim =True)
            PHI = self.c[i] * 3 /(2 * math.pi**5 * r**4)
            y = y - PHI
        loss_v = self.loss_function(self.forward(x), y)
        return loss_v

    def loss_N(self, x, y):
        outernormal = torch.ceil((x+1)/2) + torch.floor((x+1)/2) - 1
        for i in range(self.n_pred):
            g = x.clone()
            g.requires_grad = True
            xs_temp = torch.matmul(torch.ones([N_v, 1]).to(device), self.x[i:i+1])
            r = torch.sum((g - xs_temp) ** 2, dim=1, keepdim =True)
            PHI = self.c[i] * 3 /(2 * math.pi**5 * r**4)
            PHI_n = torch.sum(
                autograd.grad(PHI, g, torch.ones([N_v, 1]).to(device), retain_graph=True,
                              create_graph=True)[0] * outernormal, dim=1, keepdim=True)
            y = y - PHI_n

        g = x.clone()
        g.requires_grad = True
        v = self.forward(g)
        v_x = autograd.grad(v, g, torch.ones([N_v, 1]).to(device), retain_graph=True, create_graph=True)[0]
        yn = torch.sum(v_x * outernormal, dim = 1, keepdim=True)
        loss_v = self.loss_function(yn, y)
        return loss_v

    def loss_PDE(self, x_to_train_f):
        F = 0
        f = laplace(self.forward, x_to_train_f) + F
        loss_f = self.loss_function(f, 0*f)
        return loss_f

    def loss(self, x, y, yn, x_to_train_f, sigma):
        loss_v = self.loss_BC(x, y)
        loss_n = self.loss_N(x, yn)
        loss_f = self.loss_PDE(x_to_train_f)
        sigma_d = sigma[0,0]
        sigma_n = sigma[0,1]
        loss = sigma_d * loss_v + loss_f + sigma_n * loss_n
        return loss

    # 'callable for optimizer'
    # def closure(self):
    #     optimizer.zero_grad()
    #     loss_val = self.loss(X_u_train, u_train, un_train, X_f_train, sigma)
    #     global ite
    #     ite = ite + 1
    #     if ite % 100 == 0:
    #         error, _ = SSPINN.test()
    #         with open(log_path, "a") as f:
    #             f.write("iteration = {:d}, solution loss = {:8f}, solution error = {:8f}\n".format(ite, loss_val.item(), error.item()))
    #         sio.savemat(loss_path, {"iteration": ite, "solution loss": loss_val.item(), "solution_error": error.item()})
    #     loss_val.backward()
    #     return loss_val

    def test(self):
        v_pred = self.forward(X_u_test)
        error_vec = torch.linalg.norm((vsol - v_pred), 2) / torch.linalg.norm(vsol, 2)  # Relative L2 Norm of the error (Vector)
        # v_pred = np.reshape(v_pred.cpu().detach().numpy(), (256, 256), order='F')
        return error_vec, v_pred


# Loss Function：
# The loss function consists of two parts:
# loss_BC: MSE error of boundary losses
# loss_PDE: variational functional for the PDE

N_f = 10000
N_v = 20000
X_f_train, X_u_train, u_train, un_train = trainingdata(N_f, N_v)

noise = 0.02
u_train = u_train*(1+noise*(2*torch.rand(u_train.shape[0],u_train.shape[1]).to(device)-1))
un_train = un_train*(1+noise*(2*torch.rand(un_train.shape[0],u_train.shape[1]).to(device)-1))

layers = np.array([ndim, 50, 50, 50, 50, 50, 50, 1])

SSPINN = Sequentialmodel(layers)
SSPINN.to(device)

'Neural Network Summary'
print(SSPINN)
params = list(SSPINN.parameters())

rho = 1.2
sigma = np.matrix([20, 20])
start_time = time.time()

optimizer = optim.Adam(SSPINN.parameters(), lr=1e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer1 = optim.Adam([SSPINN.c[0],SSPINN.x[0]], lr=2e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer2 = optim.Adam([SSPINN.c[1],SSPINN.x[1]], lr=7e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer3 = optim.Adam([SSPINN.c[2],SSPINN.x[2]], lr=5e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer4 = optim.Adam([SSPINN.c[3],SSPINN.x[3]], lr=8e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer1 = optim.Adam([SSPINN.c,SSPINN.x], lr=2e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
max_iter = 80000
interval = 100
length = max_iter//interval
loss_vec = np.zeros(length)
error_vec = np.zeros(length)
iteration_vec = np.zeros(length)
# point_1 = np.zeros([length,3])
# point_2 = np.zeros([length,3])
# point_3 = np.zeros([length,3])
# point_4 = np.zeros([length,3])
k = 0
with open(log_path, "a") as f:
    f.write("Training ... \n")
for i in range(max_iter):
    loss = SSPINN.loss(X_u_train, u_train, un_train, X_f_train, sigma)
    optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
    optimizer1.zero_grad()     # zeroes the gradient buffers of all parameters
    # optimizer2.zero_grad()  # zeroes the gradient buffers of all parameters
    # optimizer3.zero_grad()     # zeroes the gradient buffers of all parameters
    # optimizer4.zero_grad()  # zeroes the gradient buffers of all parameters
    loss.backward(retain_graph=True)       #backprop
    optimizer.step()
    optimizer1.step()
    # optimizer2.step()
    # optimizer3.step()
    # optimizer4.step()
    if (i+1) % interval == 0:
        error, _ = SSPINN.test()
        with open(log_path, "a") as f:
            f.write("iteration = {:d}, solution loss = {:8f}, solution error = {:8f}\n".format(i+1, loss.item(), error.item()))
        iteration_vec[k] = i+1
        loss_vec[k] = loss.item()
        error_vec[k] = error.item()
        # point_1[k, 0] = SSPINN.c[0].item()
        # point_1[k, 1] = SSPINN.x[0, 0].item()
        # point_1[k, 2] = SSPINN.x[0, 1].item()
        # point_2[k, 0] = SSPINN.c[1].item()
        # point_2[k, 1] = SSPINN.x[1, 0].item()
        # point_2[k, 2] = SSPINN.x[1, 1].item()
        # point_3[k, 0] = SSPINN.c[2].item()
        # point_3[k, 1] = SSPINN.x[2, 0].item()
        # point_3[k, 2] = SSPINN.x[2, 1].item()
        # point_4[k, 0] = SSPINN.c[3].item()
        # point_4[k, 1] = SSPINN.x[3, 0].item()
        # point_4[k, 2] = SSPINN.x[3, 1].item()
        sio.savemat(loss_path, {"iteration": iteration_vec, "solution_loss": loss_vec, "solution_error": error_vec})
        k = k+1
        

# ite = max_iter
# while (sigma[0,0] < 200 and error_vec > 1e-2):
#     optimizer = torch.optim.LBFGS(SSPINN.parameters(), lr=0.1,
#                                   max_iter=1000,
#                                   max_eval=2500,
#                                   tolerance_grad=1e-06,
#                                   tolerance_change=1e-09,
#                                   history_size=100,
#                                   line_search_fn='strong_wolfe')
#     optimizer.zero_grad()  # zeroes the gradient buffers of all parameters
#     optimizer.step(SSPINN.closure)
#     sigma[0,0] = rho * sigma[0,0]
#     sigma[0,1] = rho * sigma[0,1]
#     error_vec, _ = SSPINN.test()

elapsed = time.time() - start_time
print('Training time: %.2f' % (elapsed))

''' Model Accuracy '''
error, u_pred = SSPINN.test()

for i in range(SSPINN.n_pred):
    xs_temp = torch.matmul(torch.ones([X_u_test.shape[0], 1]).to(device), SSPINN.x[i:i+1])
    r = torch.sum((X_u_test - xs_temp) ** 2, dim=1, keepdim = True)
    PHI = SSPINN.c[i] * 3 /(2 * math.pi**5 * r**4)
    u_pred = u_pred + PHI

sio.savemat(os.path.join(output_path, "solution.mat"), {"samples": X_u_test.cpu().detach().numpy(),
                                 "optimal_solution": usol.cpu().detach().numpy(),
                                 "pred_solution": u_pred.cpu().detach().numpy()})

# print('sigma: %f, %f' % (sigma[0,0] / rho, sigma[0,1] / rho))
print('Test Error: %.5f' % (error))

with open(log_path, "a") as f:
    f.write("\nnoise = {:f}\n" .format(noise))
for i in range(n):
    with open(log_path, "a") as f:
        f.write("point_true = {:s}, density_true ={:s}\n" .format(str(xs[i:i+1,:]), str(c[0,i])))

with open(log_path, "a") as f:
    f.write("\n")
for i in range(n):
    print('c[%d]: '%(i), SSPINN.c[i])
    print('x[%d]: '%(i), SSPINN.x[i])
    with open(log_path, "a") as f:
        f.write("point_pred ={:s}, density_pred ={:s}\n" .format(str(SSPINN.x[i].cpu().detach().numpy()), str(SSPINN.c[i].cpu().detach().numpy())))