import numpy as np
import torch
import scipy.sparse as sp
import pandas as pd
import math
import random
from sklearn.preprocessing import minmax_scale, scale
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def get_all_the_samples(A):
    m, n = A.shape
    pos = []
    neg = []
    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                pos.append([i, j, 1])
            else:
                neg.append([i, j, 0])
    n = len(pos)
    neg_new = random.sample(neg, n)
    tep_samples = pos + neg_new
    samples = random.sample(tep_samples, len(tep_samples))
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    return samples


def get_all_the_samples2(A):
    m, n = A.shape
    pos = []
    neg = []
    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                pos.append([i, j, 1])
            else:
                neg.append([i, j, 0])
    n = len(pos)
    # neg_new = random.sample(neg, n)
    tep_samples = pos + neg
    samples = random.sample(tep_samples, len(tep_samples))
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    return samples


def get_samples_PR_RWR():
    pos = np.load('./comparison/RNMFMDA/pos.npy', allow_pickle=True)
    neg = np.load('./comparison/RNMFMDA/neg.npy', allow_pickle=True)

    micRN2raw = np.load('./comparison/RNMFMDA/micRN2raw.npy', allow_pickle=True).item()
    disRN2raw = np.load('./comparison/RNMFMDA/disRN2raw.npy', allow_pickle=True).item()

    out = []
    for item in pos:
        out.append([micRN2raw[item[1]], disRN2raw[item[0]], 1])
    for item in neg:
        out.append([micRN2raw[item[1]], disRN2raw[item[0]], 0])
    out = np.array(out)
    index = list(range(out.shape[0]))
    random.shuffle(index)
    random.shuffle(index)
    out = out[index]
    return out


def get_samples_PR_RWR_biome():
    pos = np.load('./Disbiome/PU_RWR/pos.npy', allow_pickle=True)
    neg = np.load('./Disbiome/PU_RWR/neg.npy', allow_pickle=True)
    out = []
    for item in pos:
        out.append([item[1], item[0], 1])
    for item in neg:
        out.append([item[1], item[0], 0])
    out = np.array(out)
    index = list(range(out.shape[0]))
    random.shuffle(index)
    random.shuffle(index)
    out = out[index]
    return out


def get_samples_PR_RWR_peryton():
    pos = np.load('./peryton/PU_RWR/pos.npy', allow_pickle=True)
    neg = np.load('./peryton/PU_RWR/neg.npy', allow_pickle=True)
    out = []
    for item in pos:
        out.append([item[1], item[0], 1])
    for item in neg:
        out.append([item[1], item[0], 0])
    out = np.array(out)
    index = list(range(out.shape[0]))
    random.shuffle(index)
    random.shuffle(index)
    out = out[index]
    return out


def get_samples_PR_RWR_phendb():
    pos = np.load('./phendb/PU_RWR/pos.npy', allow_pickle=True)
    neg = np.load('./phendb/PU_RWR/neg.npy', allow_pickle=True)
    out = []
    for item in pos:
        out.append([item[1], item[0], 1])
    for item in neg:
        out.append([item[1], item[0], 0])
    out = np.array(out)
    index = list(range(out.shape[0]))
    random.shuffle(index)
    random.shuffle(index)
    out = out[index]
    return out


def get_syn_sim(A, k1, k2):
    disease_sim1 = np.loadtxt('./HMDAD/Dis_semantic_sim.txt')

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating(Sd_1, Sd_2, d1, d2)
    Pd_final = (Pd + Pd.T) / 2
    # np.save('d_sim_final.npy', Pd_final)
    # np.save('d_sim_final.txt', Pd_final)
    # return Pm_final, Pd_final

    return GIP_m_sim, Pd_final


def get_syn_sim2(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim1, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2
    # np.save('d_sim_final.npy', Pd_final)
    # np.save('d_sim_final.txt', Pd_final)
    # return Pm_final, Pd_final

    return GIP_m_sim, Pd_final


def get_syn_sim2_m_withNorm(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim1, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2
    # np.save('d_sim_final.npy', Pd_final)
    # np.save('d_sim_final.txt', Pd_final)
    # return Pm_final, Pd_final

    m1 = new_normalization(GIP_m_sim)

    return m1, Pd_final


def get_syn_sim3(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim1 = pd.read_csv('microbe_seq/blast_mean_sim.csv', index_col=[0]).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim1, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    m1 = new_normalization(microbe_sim1)
    m2 = new_normalization(GIP_m_sim)

    Sm_1 = KNN_kernel(microbe_sim1, k1)
    Sm_2 = KNN_kernel(GIP_m_sim, k1)

    Pm = MiRNA_updating(Sm_1, Sm_2, m1, m2)
    Pm_final = (Pm + Pm.T) / 2

    return Pm_final, Pd_final


def get_syn_sim4(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim1 = pd.read_csv('./microbe_seq/blast_mean_sim.csv', index_col=[0]).to_numpy()

    # GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim1, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    return microbe_sim1, Pd_final


def get_syn_sim4_m_withNorm(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim1 = pd.read_csv('./microbe_seq/blast_mean_sim.csv', index_col=[0]).to_numpy()

    # GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim1, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    m1 = new_normalization(microbe_sim1)

    return m1, Pd_final


def get_syn_sim_3ss_2ms(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim1 = pd.read_csv('./microbe_seq/blast_mean_sim.csv', index_col=[0]).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim1, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    m1 = new_normalization(microbe_sim1)
    # m2 = new_normalization(GIP_m_sim)
    m2 = GIP_m_sim

    Sm_1 = KNN_kernel(microbe_sim1, k1)
    Sm_2 = KNN_kernel(GIP_m_sim, k1)

    Pm = MiRNA_updating(Sm_1, Sm_2, m1, m2)
    Pm_final = (Pm + Pm.T) / 2

    return Pm_final, Pd_final


def get_syn_sim_3ss_3ms(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim1 = pd.read_csv('./microbe_seq/blast_mean_sim.csv', index_col=[0]).to_numpy()
    microbe_sim3 = mic_fun_DSS(disease_sim1, A)

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim1, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    m1 = new_normalization(microbe_sim1)
    # m2 = new_normalization(GIP_m_sim)
    m2 = GIP_m_sim
    m3 = new_normalization(microbe_sim3)

    Sm_1 = KNN_kernel(microbe_sim1, k1)
    Sm_2 = KNN_kernel(GIP_m_sim, k1)
    Sm_3 = KNN_kernel(microbe_sim3, k1)

    Pm = MiRNA_updating2(Sm_1, Sm_2, Sm_3, m1, m2, m3)
    Pm_final = (Pm + Pm.T) / 2

    return Pm_final, Pd_final


def get_syn_sim_3ss_4ms(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim1 = pd.read_csv('./microbe_seq/blast_mean_sim.csv', index_col=[0]).to_numpy()
    microbe_sim3 = mic_fun_DSS(disease_sim1, A)
    microbe_sim4 = pd.read_csv('./graph2mda/microbe_fun_sim_mdad_abio.csv', index_col=0).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim1, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    m1 = new_normalization(microbe_sim1)
    # m2 = new_normalization(GIP_m_sim)
    m2 = GIP_m_sim
    m3 = new_normalization(microbe_sim3)
    m4 = new_normalization(microbe_sim4)

    Sm_1 = KNN_kernel(microbe_sim1, k1)
    Sm_2 = KNN_kernel(GIP_m_sim, k1)
    Sm_3 = KNN_kernel(microbe_sim3, k1)
    Sm_4 = KNN_kernel(microbe_sim4, k1)

    Pm = MiRNA_updating3(Sm_1, Sm_2, Sm_3, Sm_4, m1, m2, m3, m4)
    Pm_final = (Pm + Pm.T) / 2

    return Pm_final, Pd_final


def get_syn_sim_3ss_4ms_convergeGIP(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim1 = pd.read_csv('./microbe_seq/blast_mean_sim.csv', index_col=[0]).to_numpy()
    microbe_sim3 = mic_fun_DSS(disease_sim1, A)
    microbe_sim4 = pd.read_csv('./graph2mda/microbe_fun_sim_mdad_abio.csv', index_col=0).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim1, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    m1 = new_normalization(microbe_sim1)
    # m2 = new_normalization(GIP_m_sim)
    # m2 = GIP_m_sim
    microbe_sim3 = converge_sim(GIP_m_sim, microbe_sim3)
    microbe_sim4 = converge_sim(GIP_m_sim, microbe_sim4)
    m3 = new_normalization(microbe_sim3)
    m4 = new_normalization(microbe_sim4)

    Sm_1 = KNN_kernel(microbe_sim1, k1)
    # Sm_2 = KNN_kernel(GIP_m_sim, k1)
    Sm_3 = KNN_kernel(microbe_sim3, k1)
    Sm_4 = KNN_kernel(microbe_sim4, k1)

    Pm = MiRNA_updating2(Sm_1, Sm_3, Sm_4, m1, m3, m4)
    Pm_final = (Pm + Pm.T) / 2

    return Pm_final, Pd_final


def get_syn_sim_3ss_3ms_convergeGIP(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    # microbe_sim1 = pd.read_csv('./microbe_seq/blast_mean_sim.csv', index_col=[0]).to_numpy()
    microbe_sim3 = mic_fun_DSS(disease_sim1, A)
    microbe_sim4 = pd.read_csv('./graph2mda/microbe_fun_sim_mdad_abio.csv', index_col=0).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim1, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    # m1 = new_normalization(microbe_sim1)
    # m2 = new_normalization(GIP_m_sim)
    # m2 = GIP_m_sim
    microbe_sim3 = converge_sim(GIP_m_sim, microbe_sim3)
    microbe_sim4 = converge_sim(GIP_m_sim, microbe_sim4)
    m3 = new_normalization(microbe_sim3)
    m4 = new_normalization(microbe_sim4)

    # Sm_1 = KNN_kernel(microbe_sim1, k1)
    # Sm_2 = KNN_kernel(GIP_m_sim, k1)
    Sm_3 = KNN_kernel(microbe_sim3, k1)
    Sm_4 = KNN_kernel(microbe_sim4, k1)

    Pm = MiRNA_updating(Sm_3, Sm_4, m3, m4)
    # Pm = MiRNA_updating2(Sm_1, Sm_3, Sm_4, m1, m3, m4)

    Pm_final = (Pm + Pm.T) / 2

    return Pm_final, Pd_final


def get_syn_sim_3ss_2ms_noSNF(A, k1, k2):
    disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
    disease_sim2 = pd.read_csv('./HSDN/final-symptom-disease-similarity.csv', index_col=[0]).to_numpy()

    microbe_sim1 = pd.read_csv('./microbe_seq/blast_mean_sim.csv', index_col=[0]).to_numpy()

    GIP_m_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    d1 = new_normalization(disease_sim1)
    d2 = new_normalization(disease_sim2)
    d3 = new_normalization(GIP_d_sim)
    # d1 = disease_sim1
    # d2 = disease_sim2
    # d3 = GIP_d_sim
    Sd_1 = KNN_kernel(disease_sim1, k2)
    Sd_2 = KNN_kernel(disease_sim1, k2)
    Sd_3 = KNN_kernel(GIP_d_sim, k2)
    Pd = disease_updating2(Sd_1, Sd_2, Sd_3, d1, d2, d3)
    Pd_final = (Pd + Pd.T) / 2

    m1 = new_normalization(microbe_sim1)
    # m2 = new_normalization(GIP_m_sim)
    m2 = GIP_m_sim

    return m1, m2, Pd_final


def GIP_kernel(Asso_RNA_Dis):
    # the number of row
    nc = Asso_RNA_Dis.shape[0]
    # initate a matrix as result matrix
    matrix = np.zeros((nc, nc))
    # calculate the down part of GIP fmulate
    r = getGosiR(Asso_RNA_Dis)
    # calculate the result matrix
    for i in range(nc):
        for j in range(nc):
            # calculate the up part of GIP formulate
            temp_up = np.square(np.linalg.norm(Asso_RNA_Dis[i, :] - Asso_RNA_Dis[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix


def getGosiR(Asso_RNA_Dis):
    # calculate the r in GOsi Kerel
    nc = Asso_RNA_Dis.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(Asso_RNA_Dis[i, :])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r


# W is the matrix which needs to be normalized
def new_normalization(w):
    m = w.shape[0]
    p = np.zeros([m, m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1 / 2
            elif np.sum(w[i, :]) - w[i, i] > 0:
                p[i][j] = w[i, j] / (2 * (np.sum(w[i, :]) - w[i, i]))
    return p


# get the KNN kernel, k is the number if first nearest neibors
def KNN_kernel(S, k):
    n = S.shape[0]
    S_knn = np.zeros([n, n])
    for i in range(n):
        sort_index = np.argsort(S[i, :])
        for j in sort_index[n - k:n]:
            if np.sum(S[i, sort_index[n - k:n]]) > 0:
                S_knn[i][j] = S[i][j] / (np.sum(S[i, sort_index[n - k:n]]))
    return S_knn


def disease_updating(S1, S2, P1, P2):
    it = 0
    P = (P1 + P2) / 2
    dif = 1
    while dif > 0.0000001:
        it = it + 1
        P111 = np.dot(np.dot(S1, P2), S1.T)
        P111 = new_normalization(P111)
        P222 = np.dot(np.dot(S2, P1), S2.T)
        P222 = new_normalization(P222)
        P1 = P111
        P2 = P222
        P_New = (P1 + P2) / 2
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    print("Iter numb2", it)
    return P


def disease_updating2(S1, S2, S3, P1, P2, P3):
    it = 0
    P = (P1 + P2 + P3) / 3
    dif = 1
    while dif > 0.0000001:
        it = it + 1
        P111 = np.dot(np.dot(S1, (P2 + P3) / 2), S1.T)
        P111 = new_normalization(P111)
        P222 = np.dot(np.dot(S2, (P1 + P3) / 2), S2.T)
        P222 = new_normalization(P222)
        P333 = np.dot(np.dot(S3, (P1 + P2) / 2), S3.T)
        P1 = P111
        P2 = P222
        P3 = P333
        P_New = (P1 + P2 + P3) / 3
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    print("Iter numb2", it)
    return P


# updataing rules
def MiRNA_updating(S1, S2, P1, P2):
    it = 0
    P = (P1 + P2) / 2
    dif = 1
    while dif > 0.0000001:
        it = it + 1
        P111 = np.dot(np.dot(S1, P2), S1.T)
        P111 = new_normalization(P111)
        P222 = np.dot(np.dot(S2, P1), S2.T)
        P222 = new_normalization(P222)
        P1 = P111
        P2 = P222
        P_New = (P1 + P2) / 2
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P


def MiRNA_updating2(S1, S2, S3, P1, P2, P3):
    it = 0
    P = (P1 + P2 + P3) / 3
    dif = 1
    while dif > 0.0000001:
        it = it + 1
        P111 = np.dot(np.dot(S1, (P2 + P3) / 2), S1.T)
        P111 = new_normalization(P111)
        P222 = np.dot(np.dot(S2, (P1 + P3) / 2), S2.T)
        P222 = new_normalization(P222)
        P333 = np.dot(np.dot(S3, (P1 + P2) / 2), S3.T)
        P333 = new_normalization(P333)
        P1 = P111
        P2 = P222
        P3 = P333
        P_New = (P1 + P2 + P3) / 3
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P


def MiRNA_updating3(S1, S2, S3, S4, P1, P2, P3, P4):
    it = 0
    P = (P1 + P2 + P3 + P4) / 4
    dif = 1
    while dif > 0.0000001:
        it = it + 1
        P111 = np.dot(np.dot(S1, (P2 + P3 + P4) / 3), S1.T)
        P111 = new_normalization(P111)
        P222 = np.dot(np.dot(S2, (P1 + P3 + P4) / 3), S2.T)
        P222 = new_normalization(P222)
        P333 = np.dot(np.dot(S3, (P1 + P2 + P4) / 3), S3.T)
        P333 = new_normalization(P333)
        P444 = np.dot(np.dot(S4, (P1 + P2 + P3) / 3), S3.T)
        P444 = new_normalization(P444)
        P1 = P111
        P2 = P222
        P3 = P333
        P4 = P444
        P_New = (P1 + P2 + P3 + P4) / 4
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P


def set_digo_zero(sim, z):
    sim_new = sim.copy()
    n = sim.shape[0]
    for i in range(n):
        sim_new[i][i] = z
    return sim_new


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def plot_ROC(y_true, y_pred, y_pred_proba, path='./fig/temp_ROC.tif'):
    auc_score = metrics.roc_auc_score(y_true, y_pred_proba)
    acc_score = metrics.accuracy_score(y_true, y_pred)
    pre_score = metrics.precision_score(y_true, y_pred)
    rec_score = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)

    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_proba)
    auc = metrics.auc(fpr, tpr)

    plt.figure(1)
    lw = 2

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(path, dpi=300)

    return auc_score, acc_score, pre_score, rec_score, f1_score


def plot_PR(y_true, y_pred, y_pred_proba, path='./fig/temp_PR.tif'):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred_proba)
    aupr = metrics.auc(recall, precision)

    plt.figure(1)
    lw = 2

    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='PR curve (area = %0.4f)' % aupr)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(path, dpi=300)

    return aupr


def mic_fun_DSS(Dss, Ass):
    length = Ass.shape[0]
    mfs = np.zeros((length, length))
    for k in range(length):
        for l in range(length):
            dtk = Ass[k].nonzero()[0]
            dtl = Ass[l].nonzero()[0]
            up1 = 0
            up2 = 0
            for item in dtl:
                up1 += _computeSddt(Dss, item, dtk)
            for item in dtk:
                up2 += _computeSddt(Dss, item, dtl)
            mfs[k, l] = (up1 + up2) / (len(dtl) + len(dtk))

    return mfs


def _computeSddt(Dss, d, dt):
    return Dss[d, dt].max()


def converge_sim(gip, sim):
    r, c = gip.shape
    converge = np.zeros_like(gip)
    for i in range(r):
        for j in range(c):
            if sim[i, j] != 0:
                converge[i, j] = (sim[i, j] + gip[i, j]) / 2
            else:
                converge[i, j] = gip[i, j]
    return converge


# if __name__ == '__main__':
#     A = np.load('./HMDAD/mic-dis Association.npy')
#     disease_sim1 = pd.read_csv('./HMDAD/Disease_semantic_similarity.csv', index_col=[0]).to_numpy()
#
#     temp_out = mic_fun_DSS(disease_sim1, A)


def draw_PR(folds, title=None, savePath=None, dpi=300):
    '''
    :param folds:
    :param title:
    :param savePath:
    :param dpi:
    #输入实例
        #     label = [1, 1, 0, 0,0,0,0, ] # y 是 a的值，x是各个元素的索引
        #     probability = [0.8, 0.3, 0.1, 0.2,0.3,0.4,0.6,]
        #     draw_PR({'name': [label, probability]})
    '''
    # colors = cycle(['sienna', 'seagreen', 'blue', 'red', 'darkorange', 'gold', 'orchid', 'gray'])
    c = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 使用颜色编码定义颜色
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#001e36'])

    y_real = []
    y_proba = []
    plt.figure(figsize=(8, 6))  # 指定大小分辨率等，相当于画纸
    for toolsName, y_trueANDpred_proba, color in zip(folds.keys(), folds.values(), c):
        precision, recall, thresholds = metrics.precision_recall_curve(y_trueANDpred_proba[0], y_trueANDpred_proba[1],
                                                                       pos_label=1)
        aupr = metrics.auc(recall, precision)  # 计算面积的
        plt.plot(recall, precision, color=color, label=str(toolsName) + '(AUPR = %0.4f)' % aupr,
                 lw=2, alpha=.5)  # 横纵坐标的取值，颜色样式等
        y_real.append(y_trueANDpred_proba[0])
        y_proba.append(y_trueANDpred_proba[1])
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = metrics.precision_recall_curve(y_real, y_proba)
    lab = 'Overall (AUPR=%.4f)' % (metrics.auc(recall, precision))
    plt.plot(recall, precision, color='#3333ab', label=lab, lw=2, )  # 横纵坐标的取值，颜色样式等

    plt.xlim([-0.05, 1.05])  # x坐标轴
    plt.ylim([-0.05, 1.05])  # y坐标轴
    plt.xlabel('Recall', size=12)  # x标签
    plt.ylabel('Precision', size=12)  # y标签
    # plt.legend(loc="lower left", ncol=1)
    plt.title("Precision-Recall curve")
    plt.legend(loc="lower left")
    plt.show()
    # if savePath is not None:
    #     plt.savefig(savePath, dpi=dpi)
    # plt.close()


def draw_10fold_ROC(label=None, score=None):
    #############################################################
    # GATMDA
    data = np.load('F:/VGAE/comparison/GATMDA-master/src/temp/10K_label_score.npy').reshape(20, -1)
    label = data[:10, :]
    score = data[10:, :]
    #############################################################
    # 10 fold
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(10):
        fpr, tpr, _ = metrics.roc_curve(label[i], score[i])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, label='{} fold (auc = {:.4f})'.format(i, roc_auc))
    plt.plot([0, 1], [0, 1], lw=2, c='r', label='Random', linestyle='--')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey',
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
