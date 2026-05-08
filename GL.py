import numpy as np
import pandas as pd
import datetime
import random
import string
import pickle
import os
from dataclasses import dataclass

from sklearn.covariance import graphical_lasso
from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.helper.data_generation import time_varying_power_network, sample_covariance_matrix
from gglasso.helper.experiment_helper import discovery_rate, error
from gglasso.helper.utils import get_K_identity
from gglasso.helper.experiment_helper import plot_evolution, plot_deviation, surface_plot, single_heatmap_animation
from gglasso.helper.model_selection import aic, ebic

@dataclass
class NetworkData:
    time: datetime.date
    jaccard_index: float
    A: np.ndarray
    Theta: np.ndarray
    l1_penalty: float


def select_best_sgl(S, N, gamma=0.1):
    K, p, _ = S.shape
    alphas = 2 * np.logspace(2, -2, 8)
    bic_vals = np.zeros(len(alphas))
    Theta_list = []
    valid_idx = [k for k in range(K) if not np.any(np.isnan(S[k]))]
    if len(valid_idx) < K:
        print(f"发现 {K - len(valid_idx)} 个含 NaN 的协方差矩阵")
        S = S[valid_idx]
        K = len(valid_idx)

    for i, alpha in enumerate(alphas):
        Theta_tmp = np.zeros((K, p, p))
        for k in range(K):
            _, prec = graphical_lasso(S[k], alpha=alpha,max_iter=10000,tol=5e-3)
            Theta_tmp[k] = prec
        bic_vals[i] = ebic(S, Theta_tmp, N, gamma=gamma)
        Theta_list.append(Theta_tmp.copy())

    best_idx = np.nanargmin(bic_vals)
    print(f"SGL 最佳 alpha: {alphas[best_idx]:.5f}")
    print(f"SGL 最佳 ebic: {bic_vals[best_idx]:.5f}")
    return alphas[best_idx], Theta_list[best_idx]



def select_best_fgl(S, N, gamma=0.1, use_aic=False):
    K, p, _ = S.shape
    l1_candidates = 2 * np.logspace(2, -2, 8)  
    l2_candidates = 2 * np.logspace(2, -2, 8)

    L2, L1 = np.meshgrid(l2_candidates, l1_candidates)  
    n1, n2 = L1.shape

    AIC_mat = np.zeros((n1, n2))
    BIC_mat = np.zeros((n1, n2))

    Omega_0 = get_K_identity(K, p)
    Theta_0 = get_K_identity(K, p)
    X_0 = np.zeros((K, p, p))

    for g1 in range(n1):
        for g2 in range(n2):
            lam1 = L1[g1, g2]
            lam2 = L2[g1, g2]
            try:
                sol, info = ADMM_MGL(
                    S, lam1, lam2, 'FGL',
                    Omega_0,
                    Theta_0=Theta_0,
                    X_0=X_0,
                    tol=1e-5, rtol=1e-5,
                    verbose=False, measure=False,max_iter=2000
                )
                Theta_sol = sol['Theta']
                # warm start
                Omega_0 = (sol['Omega'] + sol['Omega'].transpose(0,2,1)) / 2.0
                Theta_0 = (Theta_sol + Theta_sol.transpose(0,2,1)) / 2.0
                X_0 = (sol['X'] + sol['X'].transpose(0,2,1)) / 2.0
                #防止warm start不对称
                AIC_mat[g1, g2] = aic(S, Theta_sol, N)
                BIC_mat[g1, g2] = ebic(S, Theta_sol, N, gamma=gamma)
            except np.linalg.LinAlgError:
                AIC_mat[g1, g2] = np.inf
                BIC_mat[g1, g2] = np.inf
                continue

    if use_aic:
        ix = np.unravel_index(np.nanargmin(AIC_mat), AIC_mat.shape)
    else:
        ix = np.unravel_index(np.nanargmin(BIC_mat), BIC_mat.shape)

    lam1_opt = L1[ix]
    lam2_opt = L2[ix]
    print(f"FGL 最佳 (alpha, beta): ({lam1_opt:.5f}, {lam2_opt:.5f})")
    print(f"FGL 最佳 eBIC: {BIC_mat[ix]:.5f}")

    sol_opt, _ = ADMM_MGL(
        S, lam1_opt, lam2_opt, 'FGL',
        get_K_identity(K, p),
        Theta_0=get_K_identity(K, p),
        X_0=np.zeros((K, p, p)),
        tol=1e-5, rtol=1e-5, verbose=False
    )
    return lam1_opt, lam2_opt, sol_opt['Theta']


def compute_l1_distance(Theta_seq):
    K = len(Theta_seq)
    l1 = [0.0]
    for t in range(1, K):
        diff = Theta_seq[t] - Theta_seq[t-1]
        l1.append(np.sum(np.abs(diff)))
    return np.array(l1)


def threshold_to_adjacency(Theta_seq, threshold):
    adj_seq = []
    for Theta in Theta_seq:
        adj = (np.abs(Theta) >= threshold).astype(int)
        np.fill_diagonal(adj, 0)
        adj_seq.append(adj)
    return np.stack(adj_seq)


def jaccard_index_sequence(adj_seq):
    K = len(adj_seq)
    jacc = [1.0]
    for t in range(1, K):
        prev = adj_seq[t-1].astype(bool)
        curr = adj_seq[t].astype(bool)
        inter = np.sum(prev & curr)
        union = np.sum(prev | curr)
        if union == 0:
            jacc.append(1.0)
        else:
            jacc.append(inter / union)
    return np.array(jacc)


def simulate_stock_experiment(N_assets=8, T=20, method='SGL',
                              output_file='simulated_data.pkl', threshold=0.1):
    p = N_assets
    K = T
    M = 3
    if p % M != 0:
        p = (p // M) * M
        if p == 0:
            p = M
        print(f"注意：资产数调整为 {p} 以满足 M={M} 的整除要求。")

    N_samples = 1000

    Sigma, Theta_true = time_varying_power_network(p, K, M, scale=False, seed=42)
    S, _ = sample_covariance_matrix(Sigma, N_samples)

    if method == 'SGL':
        _, Theta_est = select_best_sgl(S, N_samples, gamma=0.1)
    elif method == 'FGL':
        _, _, Theta_est = select_best_fgl(S, N_samples, gamma=0.1)
    else:
        raise ValueError("method 必须为 'SGL' 或 'FGL'")

    adj_seq = threshold_to_adjacency(Theta_est, threshold)
    l1_dist = compute_l1_distance(Theta_est)        
    jaccard_seq = jaccard_index_sequence(adj_seq)

    # 随机生成资产简称
    names = []
    for _ in range(p):
        length = random.randint(2, 4)
        name = ''.join(random.choices(string.ascii_uppercase, k=length))
        names.append(name)

    start_date = datetime.date(2020, 1, 1)
    times = [start_date + pd.DateOffset(months=i) for i in range(K)]
    times = [t.date() for t in times]

    # 封装为 NetworkData 序列
    data_array = []
    for idx, t in enumerate(times):
        data_array.append(NetworkData(
            time=t,
            jaccard_index=jaccard_seq[idx],
            A=adj_seq[idx].copy(),
            Theta=Theta_est[idx].copy(),
            l1_penalty=l1_dist[idx] 
        ))

    with open(output_file, 'wb') as f:
        pickle.dump({'name': names, 'data_array': data_array}, f)
    print(f"结果已保存至 {output_file}")


if __name__ == '__main__':
    #example:模拟生成两个网络，模仿GGLASSO中的FGL样例
    simulate_stock_experiment(N_assets=30, T=24, method='SGL',
                              output_file='stock_sgl.pkl', threshold=0.05)
    
    # 也可生成 FGL 估计的结果
    simulate_stock_experiment(N_assets=30, T=24, method='FGL',
                              output_file='stock_fgl.pkl', threshold=0.05)