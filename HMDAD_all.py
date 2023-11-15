import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, precision_score, roc_auc_score, plot_roc_curve
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import scipy.sparse as sp
from torch.optim import Adam

from layers import SinkhornDistance
from model3 import *
from utils import *
from get_sim import *
import args
import os
import time
import matplotlib.pyplot as plt
import random

torch.manual_seed(1)
random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# paprameters
k1 = 30
k2 = 5
D = 90  # MF dimension
A = np.load('./HMDAD/mic-dis Association.npy')
print("the number of miRNAs and diseases", A.shape)
print("the number of associations", sum(sum(A)))
x, y = A.shape
score_matrix = np.zeros([x, y])
samples = get_all_the_samples(A)

sim_m, sim_d = get_3ss_GIP_dis_drug(A, k1, k2)
sim_m_0 = set_digo_zero(sim_m, 0)
sim_d_0 = set_digo_zero(sim_d, 0)

features_m = sparse_to_tuple(sp.coo_matrix(A))
features_d = sparse_to_tuple(sp.coo_matrix(A.transpose()))

w_aux1 = 0.05
w_aux2 = 0.05
w_w = 0.90

suffix = '3ss_blastn_convergeGIP_dis_drug_ms_' + str(w_aux1) + '_' + str(w_aux2) + '_' + str(w_w)
# suffix = 'base_1z_kl'
##########################################################################################
# 对微生物相似性

adj_norm = preprocess_graph(sim_m_0)

adj = sim_m_0
# pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
# norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
pos_weight = 25
norm = 0.5

sim_m_0 = sp.coo_matrix(sim_m_0)
sim_m_0.eliminate_zeros()
adj_label = sim_m_0 + sp.eye(sim_m_0.shape[0])
adj_label = sparse_to_tuple(adj_label)

adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                    torch.FloatTensor(adj_norm[1]),
                                    torch.Size(adj_norm[2])).to(device)
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                     torch.FloatTensor(adj_label[1]),
                                     torch.Size(adj_label[2])).to_dense().to(device)
features = torch.sparse.FloatTensor(torch.LongTensor(features_m[0].T),
                                    torch.FloatTensor(features_m[1]),
                                    torch.Size(features_m[2])).to(device)

weight_mask = adj_label.view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0)).to(device)
# weight_tensor[weight_mask] = pos_weight
weight_tensor[weight_mask] = 25

# init model and optimizer
model = VGAE3()
model.to(device)
print(model)

# optimizer = Adam(model.parameters(), lr=args.learning_rate)
optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
print("Optimizer", optimizer)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.8, last_epoch=-1)

sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, device=device)

tra_auc, tra_ap, tra_l, v_auc, v_ap = [], [], [], [], []
tra_acc, tra_baseloss, tra_aux1, tra_aux2 = [], [], [], []
train_wloss, tra_kl = [], []
tra_r2, tra_RMSE = [], []

min_loss = 10.
min_mse = 100.

for epoch in range(5000):
    t = time.time()
    model.train()
    A_pred, x1_pred, x2_pred, x1, x2, z = model(adj_norm, features)

    optimizer.zero_grad()
    loss = base_loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1), weight=weight_tensor)
    # print('#######################')
    # print('base_loss:', loss.item())
    tra_baseloss.append(loss.item())

    kl_divergence = 0.5 / A_pred.size(0) * (
            1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2).sum(1).mean()
    tra_kl.append(-kl_divergence.item())
    # loss -= kl_divergence
    # loss -= kl_divergence * w_w

    loss_aux1 = norm * F.binary_cross_entropy(x1_pred.view(-1), x1.detach().view(-1))
    tra_aux1.append(loss_aux1.item())
    loss += loss_aux1 * w_aux1
    # loss += loss_aux1

    loss_aux2 = norm * F.binary_cross_entropy(x2_pred.view(-1), x2.detach().view(-1))
    tra_aux2.append(loss_aux2.item())
    loss += loss_aux2 * w_aux2
    # loss += loss_aux2

    wasser_loss = 0.5 / A_pred.size(0) * sinkhorn(z, torch.randn_like(z))[0]
    train_wloss.append(wasser_loss.item())
    loss += wasser_loss * w_w
    # loss += wasser_loss

    loss.backward()
    optimizer.step()
    scheduler.step()

    r2 = r2_score(adj_label.cpu().detach().numpy().reshape(-1), A_pred.cpu().detach().numpy().reshape(-1))
    RMSE = np.sqrt(mean_squared_error(adj_label.cpu().detach().numpy().reshape(-1), A_pred.cpu().detach().numpy().reshape(-1)))
    tra_r2.append(r2)
    tra_RMSE.append(RMSE)

    tra_l.append(loss.item())
    print('--------------------------------')
    print("Epoch:", '%04d' % (epoch + 1), "base_loss=", "{:.5f}".format(base_loss.item()),
          "train_loss=", "{:.5f}".format(loss.item()),
          "train_r2=", "{:.5f}".format(r2), "train_RMSE=", "{:.5f}".format(RMSE),
          "time=", "{:.5f}".format(time.time() - t))
    if RMSE < min_mse:
        min_mse = RMSE
        state = {'model': model.state_dict(),
                 'epoch': epoch,
                 'min_mse': min_mse,
                 }
        torch.save(state, 'models1/M_mse_256_128_64_5000_' + suffix + '.pth')

model.eval()

save_dic = {'train_loss': tra_l,
            'train_baseloss': tra_baseloss,
            'train_aux1': tra_aux1,
            'train_aux2': tra_aux2,
            'train_wloss': train_wloss,
            'train_kl': tra_kl,
            'train_r2': tra_r2,
            'train_RMSE': tra_RMSE,
            }
np.save('./metrics1/M_mse_256_128_64_5000_' + suffix + '.npy', save_dic)
model_ = VGAE3()
model_.to(device)
model_.load_state_dict(torch.load('models1/M_mse_256_128_64_5000_' + suffix + '.pth')['model'])
model_.eval()
out_m = model_(adj_norm, features)

latent_m = out_m[-1].cpu().detach().numpy()
np.save('./latent/latent_m_mse_' + suffix + '.npy', latent_m)

##########################################################################################
# 对疾病相似性

adj_norm2 = preprocess_graph(sim_d_0)

adj2 = sim_d_0
# pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
# norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
pos_weight2 = 25
norm2 = 0.5

sim_d_0 = sp.coo_matrix(sim_d_0)
sim_d_0.eliminate_zeros()
adj_label2 = sim_d_0 + sp.eye(sim_d_0.shape[0])
adj_label2 = sparse_to_tuple(adj_label2)

adj_norm2 = torch.sparse.FloatTensor(torch.LongTensor(adj_norm2[0].T),
                                     torch.FloatTensor(adj_norm2[1]),
                                     torch.Size(adj_norm2[2])).to(device)
adj_label2 = torch.sparse.FloatTensor(torch.LongTensor(adj_label2[0].T),
                                      torch.FloatTensor(adj_label2[1]),
                                      torch.Size(adj_label2[2])).to_dense().to(device)
features2 = torch.sparse.FloatTensor(torch.LongTensor(features_d[0].T),
                                     torch.FloatTensor(features_d[1]),
                                     torch.Size(features_d[2])).to(device)

weight_mask2 = adj_label2.view(-1) == 1
weight_tensor2 = torch.ones(weight_mask2.size(0)).to(device)
# weight_tensor[weight_mask] = pos_weight
weight_tensor2[weight_mask2] = 25

# init model and optimizer
model2 = VGAE4()
model2.to(device)

# optimizer2 = Adam(model2.parameters(), lr=0.001)
optimizer2 = Adam(model2.parameters(), lr=0.001, weight_decay=1e-4)
print("Optimizer", optimizer2)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, 500, gamma=0.8, last_epoch=-1)

sinkhorn2 = SinkhornDistance(eps=0.1, max_iter=100, device=device)

tra_auc, tra_ap, tra_l, v_auc, v_ap = [], [], [], [], []
tra_acc, tra_baseloss, tra_aux1, tra_aux2 = [], [], [], []
train_wloss, tra_kl = [], []
tra_r2, tra_RMSE = [], []

max_r2 = 0.
min_loss = 10.
min_mse = 100.

for epoch in range(5000):
    t = time.time()
    model2.train()
    A_pred, x1_pred, x2_pred, x1, x2, z = model2(adj_norm2, features2)

    optimizer2.zero_grad()

    loss = base_loss = norm2 * F.binary_cross_entropy(A_pred.view(-1), adj_label2.view(-1), weight=weight_tensor2)
    # loss = base_loss = norm2 * torch.sqrt(F.mse_loss(A_pred.view(-1), adj_label2.view(-1)))
    # print('#######################')
    # print('base_loss:', loss.item())
    tra_baseloss.append(loss.item())

    kl_divergence = 0.5 / A_pred.size(0) * (
            1 + 2 * model2.logstd - model2.mean ** 2 - torch.exp(model2.logstd) ** 2).sum(1).mean()
    tra_kl.append(-kl_divergence.item())
    # loss -= kl_divergence * w_w
    # loss -= kl_divergence

    loss_aux1 = norm * F.binary_cross_entropy(x1_pred.view(-1), x1.detach().view(-1))
    tra_aux1.append(loss_aux1.item())
    loss += loss_aux1 * w_aux1
    # loss += loss_aux1

    loss_aux2 = norm * F.binary_cross_entropy(x2_pred.view(-1), x2.detach().view(-1))
    tra_aux2.append(loss_aux2.item())
    loss += loss_aux2 * w_aux2
    # loss += loss_aux2

    wasser_loss = 0.5 / A_pred.size(0) * sinkhorn2(z, torch.randn_like(z))[0]
    train_wloss.append(wasser_loss.item())
    loss += wasser_loss * w_w
    # loss += wasser_loss

    loss.backward()
    optimizer2.step()
    scheduler2.step()

    r2 = r2_score(adj_label2.cpu().detach().numpy().reshape(-1), A_pred.cpu().detach().numpy().reshape(-1))
    RMSE = np.sqrt(mean_squared_error(adj_label2.cpu().detach().numpy().reshape(-1), A_pred.cpu().detach().numpy().reshape(-1)))
    tra_r2.append(r2)
    tra_RMSE.append(RMSE)

    tra_l.append(loss.item())
    print('--------------------------------')
    print("Epoch:", '%04d' % (epoch + 1), "base_loss=", "{:.5f}".format(base_loss.item()),
          "train_loss=", "{:.5f}".format(loss.item()),
          "train_r2=", "{:.5f}".format(r2), "train_RMSE=", "{:.5f}".format(RMSE),
          "time=", "{:.5f}".format(time.time() - t))
    if base_loss.item() < min_loss:
        min_loss = base_loss.item()
        state = {'model': model2.state_dict(),
                 'epoch': epoch,
                 'min_loss': min_loss,
                 }
        torch.save(state, 'models1/D_loss_256_128_64_5000_' + suffix + '.pth')
model2.eval()

save_dic = {'train_loss': tra_l,
            'train_baseloss': tra_baseloss,
            'train_aux1': tra_aux1,
            'train_aux2': tra_aux2,
            'train_wloss': train_wloss,
            'train_kl': tra_kl,
            'train_r2': tra_r2,
            'train_RMSE': tra_RMSE,
            }
np.save('./metrics1/D_loss_256_128_64_5000_' + suffix + '.npy', save_dic)

model2_ = VGAE4()
model2_.to(device)
model2_.load_state_dict(torch.load('models1/D_loss_256_128_64_5000_' + suffix + '.pth')['model'])
model2_.eval()
out_d = model2_(adj_norm2, features2)

latent_d = out_d[-1].cpu().detach().numpy()
np.save('./latent/latent_d_loss_' + suffix + '.npy', latent_d)

