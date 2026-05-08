from GL_anl import (
    extract_cov_sequence,
    analyze_portfolio
)
from GL import (
    select_best_fgl,
    select_best_sgl,
    threshold_to_adjacency,
    jaccard_index_sequence,
    compute_l1_distance,
    NetworkData
)
import numpy as np

#analyze_portfolio(txt_file='macro_30.txt',h5_file='stock_daily_data_1.h5', token='b3189fe9b3f0ddc26093c08ed4a0b76a104d29f34eb049eaf1f511d8',output_txt='portfolio_analysis_output.txt',
#                  start_date='20060101', end_date='20260501')

#使用前先用analyze_portfolio回看一下数据，手动选择无数据缺失的连续时间

W = 2 #窗口长度
N = 20 * W
name,S,date=extract_cov_sequence(txt_file='macro_30.txt',h5_file='stock_daily_data_1.h5', token='your token here',
                     start_date='20180601', end_date='20260501',factor=8000,W=W)

#这里提取了macro_30.txt的代码组合，并将S序列乘8000

_,Theta = select_best_sgl(S, N, gamma=0.1) #或者是_,_,Theta = select_best_fgl(S, N, gamma=0.1)

l1_pen = compute_l1_distance(Theta)
Adj_seq = threshold_to_adjacency(Theta, 1e-2)
jaccard_seq = jaccard_index_sequence(Adj_seq)

from datetime import datetime
date_list = [datetime.strptime(str(m), '%Y-%m').date() for m in date]

data_array = []
for idx, t in enumerate(date_list): 
    data_array.append(NetworkData(
        time=t,
        jaccard_index=jaccard_seq[idx],
        A=Adj_seq[idx].copy(),
        Theta=Theta[idx].copy(),
        l1_penalty=l1_pen[idx]
    ))

import pickle

#保存到SGL_M3_W2.pkl

with open('SGL_M3_W2.pkl', 'wb') as f:
    pickle.dump({'name': name, 'data_array': data_array}, f)