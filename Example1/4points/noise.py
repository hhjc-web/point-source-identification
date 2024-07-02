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
import os
import scipy.io as sio
from cumpute_number import compute_number
from typing import Union, Dict, Optional, Tuple, List
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

# Data Prep
N = 501
x_1 = np.linspace(-1, 1, N)
x_2 = np.linspace(1, -1, N)
X, Y = np.meshgrid(x_1, x_2)

# Test Data

X_u_test = np.hstack((X.flatten(order='F')[:, None], Y.flatten(order='F')[:, None]))
lb = np.array([-1, -1])  # lower bound
ub = np.array([1, 1])  # upper bound
# source term
c = np.matrix([1,2,3,4])
x = np.matrix([[1/2,1/2],[-1/2,1/2],[-1/2,-1/2],[1/2,-1/2]])

PHI = [0,0,0,0]
PHI_x = [0,0,0,0]
PHI_y = [0,0,0,0]

n = 4

for i in range(n):
    PHI[i] = -c[0,i] / (4 * np.pi) * np.log((X-x[i,0]) ** 2 + (Y-x[i,1]) ** 2)
    PHI_x[i] = -c[0,i] / (2*np.pi) * (X-x[i,0])/((X-x[i,0]) ** 2 + (Y-x[i,1]) ** 2)
    PHI_y[i] = -c[0,i] / (2*np.pi) * (Y-x[i,1])/((X-x[i,0]) ** 2 + (Y-x[i,1]) ** 2)

vsol = X**2 - Y**2
vsol_x = 2 * X
vsol_y = -2 * Y


usol = vsol
ux = vsol_x
uy = vsol_y
for i in range(n):
    usol = usol + PHI[i]
    ux = ux + PHI_x[i]
    uy = uy + PHI_y[i]

v_true = vsol.flatten('F')[:, None]
u_true = usol.flatten('F')[:, None]

# Training Data

def trainingdata(N_f):
    leftedge_x = np.hstack((X[:, 0][:, None], Y[:, 0][:, None]))
    leftedge_u = usol[:, 0][:, None]
    leftdiff_u = -ux[:, 0][:, None]

    rightedge_x = np.hstack((X[:, -1][:, None], Y[:, -1][:, None]))
    rightedge_u = usol[:, -1][:, None]
    rightdiff_u = ux[:, -1][:, None]

    topedge_x = np.hstack((X[0, :][:, None], Y[0, :][:, None]))
    topedge_u = usol[0, :][:, None]
    topdiff_u = uy[0, :][:, None]

    bottomedge_x = np.hstack((X[-1, :][:, None], Y[-1, :][:, None]))
    bottomedge_u = usol[-1, :][:, None]
    bottomdiff_u = -uy[-1, :][:, None]

    all_X_u_train = np.vstack([leftedge_x, rightedge_x, bottomedge_x, topedge_x])
    all_u_train = np.vstack([leftedge_u, rightedge_u, bottomedge_u, topedge_u])
    all_un_train = np.vstack([leftdiff_u, rightdiff_u, bottomdiff_u, topdiff_u])

    # choose random N_v points for training
    # idx = np.random.choice(all_X_u_train.shape[0], N_v, replace=False)
    #
    # X_u_train = all_X_u_train[idx[0:N_v], :]  # choose indices from  set 'idx' (x,t)
    # u_train = all_u_train[idx[0:N_v], :]  # choose corresponding v
    # un_train = all_un_train[idx[0:N_v], :]  # choose corresponding vn

    X_u_train = all_X_u_train
    u_train = all_u_train
    un_train = all_un_train

    '''Collocation Points'''
    # N_f sets of tuples(x,t)
    X_f = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f, X_u_train))  # append training points to collocation points
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
        self.n_pred = compute_number(X_u_train, u_train, un_train, N, M=8)
        with open(log_path, "w") as f:
            f.write("Number of sources:{:d} \n" .format(self.n_pred))
        rand_point = 2*torch.rand(self.n_pred,2) - 1
        self.x = torch.autograd.Variable(rand_point.to(device), requires_grad=True)
        self.c = torch.autograd.Variable(torch.rand(self.n_pred,1).to(device), requires_grad=True)
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
            PHI = -self.c[i] / (4 * np.pi) * torch.log((x[:,0:1] - self.x[i][0]) ** 2 + (x[:,1:2] - self.x[i][1]) ** 2)
            y = y - PHI
        loss_v = self.loss_function(self.forward(x), y)
        return loss_v

    def loss_N(self, x, y):
        outernormal = torch.ceil((x+1)/2) + torch.floor((x+1)/2) - 1
        for k in range(4):
            temp = outernormal[k * N:(k + 1) * N, :]
            temp[0] = temp[1]
            temp[-1] = temp[1]
        for i in range(self.n_pred):
            PHI = -self.c[i] / (4 * np.pi) * torch.log((x[:,0:1] - self.x[i][0]) ** 2 + (x[:,1:2] - self.x[i][1]) ** 2)
            PHI_x = -self.c[i] / (2 * np.pi) * (x[:,0:1] - self.x[i][0]) / ((x[:,0:1] - self.x[i][0]) ** 2 + (x[:,1:2] - self.x[i][1]) ** 2)
            PHI_y = -self.c[i] / (2 * np.pi) * (x[:,1:2] - self.x[i][1]) / ((x[:,0:1] - self.x[i][0]) ** 2 + (x[:,1:2] - self.x[i][1]) ** 2)
            gradPHI = torch.hstack((PHI_x, PHI_y))
            PHI_n = torch.sum(gradPHI * outernormal, dim = 1, keepdim=True)
            y = y - PHI_n
        g = x.clone()
        g.requires_grad = True
        v = self.forward(g)
        v_x = autograd.grad(v, g, torch.ones([x.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
        grady = torch.hstack((v_x[:, [0]],v_x[:, [1]]))
        yn = torch.sum(grady * outernormal, dim = 1, keepdim=True)
        loss_v = self.loss_function(yn, y)
        return loss_v

    def loss_PDE(self, x_to_train_f):
        x_1_f = x_to_train_f[:, [0]]
        x_2_f = x_to_train_f[:, [1]]
        g = x_to_train_f.clone()
        g.requires_grad = True
        v = self.forward(g)
        v_x = autograd.grad(v, g, torch.ones([x_to_train_f.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
        v_x_1 = v_x[:, [0]]
        v_x_2 = v_x[:, [1]]
        F = 0

        # f = 1 / 2 * (v_x_1 ** 2 + v_x_2 ** 2) - F * v
        # loss_f = torch.mean(f)

        v_xx_1 = autograd.grad(v_x_1, g, torch.ones([x_to_train_f.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
        v_xx_2 = autograd.grad(v_x_2, g, torch.ones([x_to_train_f.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
        v_xx_1 = v_xx_1[:, [0]]
        v_xx_2 = v_xx_2[:, [1]]
        f = v_xx_1 + v_xx_2 + F
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
        v_pred = self.forward(X_u_test_tensor)
        error_vec = torch.linalg.norm((v - v_pred), 2) / torch.linalg.norm(v, 2)  # Relative L2 Norm of the error (Vector)
        # v_pred = np.reshape(v_pred.cpu().detach().numpy(), (256, 256), order='F')
        return error_vec, v_pred


# Loss Function：
# The loss function consists of two parts:
# loss_BC: MSE error of boundary losses
# loss_PDE: variational functional for the PDE

N_f = 3000
X_f_train_np_array, X_u_train_np_array, u_train_np_array, un_train_np_array = trainingdata(N_f)

noise = 0.02
u_train_np_array = u_train_np_array*(1+noise*(2*np.random.rand(u_train_np_array.shape[0],u_train_np_array.shape[1])-1))
un_train_np_array = un_train_np_array*(1+noise*(2*np.random.rand(un_train_np_array.shape[0],u_train_np_array.shape[1])-1))

'Convert to tensor and send to GPU'
X_f_train = torch.from_numpy(X_f_train_np_array).float().to(device)
X_u_train = torch.from_numpy(X_u_train_np_array).float().to(device)
u_train = torch.from_numpy(u_train_np_array).float().to(device)
un_train = torch.from_numpy(un_train_np_array).float().to(device)
X_u_test_tensor = torch.from_numpy(X_u_test).float().to(device)
v = torch.from_numpy(v_true).float().to(device)

layers = np.array([2, 20, 20, 1])

SSPINN = Sequentialmodel(layers)
SSPINN.to(device)

'Neural Network Summary'
print(SSPINN)
params = list(SSPINN.parameters())

rho = 1.2
sigma = np.matrix([10, 10])
start_time = time.time()

optimizer = optim.Adam(SSPINN.parameters(), lr=1e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer1 = optim.Adam([SSPINN.c[0],SSPINN.x[0]], lr=2e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer2 = optim.Adam([SSPINN.c[1],SSPINN.x[1]], lr=7e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer3 = optim.Adam([SSPINN.c[2],SSPINN.x[2]], lr=5e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer4 = optim.Adam([SSPINN.c[3],SSPINN.x[3]], lr=8e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer1 = optim.Adam([SSPINN.c,SSPINN.x], lr=6e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
max_iter = 10000
interval = 100
length = max_iter//interval
loss_vec = np.zeros(length)
error_vec = np.zeros(length)
iteration_vec = np.zeros(length)
point_1 = np.zeros([length,3])
point_2 = np.zeros([length,3])
point_3 = np.zeros([length,3])
point_4 = np.zeros([length,3])
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
    loss.backward()       #backprop
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
        point_1[k, 0] = SSPINN.c[0].item()
        point_1[k, 1] = SSPINN.x[0, 0].item()
        point_1[k, 2] = SSPINN.x[0, 1].item()
        point_2[k, 0] = SSPINN.c[1].item()
        point_2[k, 1] = SSPINN.x[1, 0].item()
        point_2[k, 2] = SSPINN.x[1, 1].item()
        point_3[k, 0] = SSPINN.c[2].item()
        point_3[k, 1] = SSPINN.x[2, 0].item()
        point_3[k, 2] = SSPINN.x[2, 1].item()
        point_4[k, 0] = SSPINN.c[3].item()
        point_4[k, 1] = SSPINN.x[3, 0].item()
        point_4[k, 2] = SSPINN.x[3, 1].item()
        sio.savemat(loss_path, {"iteration": iteration_vec, "solution_loss": loss_vec, "solution_error": error_vec,
                                "point_1": point_1, "point_2": point_2, "point_3": point_3, "point_4": point_4})
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
    PHIi = -SSPINN.c[i] / (4 * np.pi) * torch.log((X_u_test_tensor[:,0:1]- SSPINN.x[i][0]) ** 2 + (X_u_test_tensor[:,1:2] - SSPINN.x[i][1]) ** 2)
    u_pred = u_pred + PHIi

sio.savemat(os.path.join(output_path, "solution.mat"), {"samples": X_u_test,
                                 "optimal_solution": u_true,
                                 "pred_solution": u_pred.detach().numpy()})

# print('sigma: %f, %f' % (sigma[0,0] / rho, sigma[0,1] / rho))
print('Test Error: %.5f' % (error))

with open(log_path, "a") as f:
    f.write("\nnoise = {:f}\n" .format(noise))
for i in range(n):
    with open(log_path, "a") as f:
        f.write("point_true = {:s}, density_true ={:s}\n" .format(str(x[i,:]), str(c[0,i])))

with open(log_path, "a") as f:
    f.write("\n")
for i in range(n):
    print('c[%d]: '%(i), SSPINN.c[i])
    print('x[%d]: '%(i), SSPINN.x[i])
    with open(log_path, "a") as f:
        f.write("point_pred ={:s}, density_pred ={:s}\n" .format(str(SSPINN.x[i].detach().numpy()), str(SSPINN.c[i].detach().numpy())))