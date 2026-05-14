import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from GL import (
    select_best_fgl,
    select_best_sgl,
    jaccard_index_sequence,
    compute_l1_distance,
    NetworkData
)

def cov_to_corr(S):
    """
    将协方差矩阵序列 S (shape: K x p x p) 逐矩阵转换为相关系数矩阵。
    若某矩阵对角线出现非正值，发出警告并将对应行/列置为 NaN。
    """
    S = np.asarray(S)
    if S.ndim != 3:
        raise ValueError(f"期望 S 为 3 维数组 (K, p, p)，实际得到 {S.ndim} 维")

    K, p, _ = S.shape
    corr_seq = np.empty_like(S)

    for k in range(K):
        cov = S[k]
        diag = np.diag(cov)
        bad = diag <= 0
        if np.any(bad):
            bad_indices = np.where(bad)[0]
            print(f"警告: 第 {k} 个矩阵对角线含非正值，索引: {bad_indices.tolist()}")
            diag_sqrt = np.sqrt(np.maximum(diag, 0))
            corr = cov / np.outer(diag_sqrt, diag_sqrt)
            corr[bad, :] = np.nan
            corr[:, bad] = np.nan
        else:
            diag_sqrt = np.sqrt(diag)
            corr = cov / np.outer(diag_sqrt, diag_sqrt)
        corr_seq[k] = corr

    return corr_seq


def analyze_portfolio(txt_file,
                      start_date=None,
                      end_date=None,
                      h5_file='stock_daily_data.h5',
                      output_txt=None):
    output_buffer = []

    def log(msg):
        print(msg)
        output_buffer.append(msg)

    with open(txt_file, 'r') as f:
        codes = [line.strip() for line in f if line.strip()]
    if not codes:
        log("资产列表为空，请检查txt文件。")
        return

    # 从 h5 文件读取资产简写
    try:
        name_map_df = pd.read_hdf(h5_file, key='meta/name_map')
        name_map = dict(zip(name_map_df['ts_code'], name_map_df['cnspell']))
    except (KeyError, Exception):
        name_map = {code: code for code in codes}

    log("========== 资产简写（cnspell） ==========")
    for code in codes:
        log(f"{code} : {name_map.get(code, '未知')}")
    log("========================================\n")

    ret_series = {}
    for code in codes:
        key = f'data/{code.replace(".", "_")}'  
        try:
            df = pd.read_hdf(h5_file, key=key)
        except KeyError:
            log(f"警告: 在 {h5_file} 中未找到 {code} 的数据 (key={key})，已跳过。")
            continue
        except Exception as e:
            log(f"读取 {code} 数据失败: {e}")
            continue

        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.sort_values('trade_date')
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df_ret = df[['trade_date', 'log_ret']].dropna()
        df_ret = df_ret.set_index('trade_date')['log_ret']
        df_ret.name = code
        ret_series[code] = df_ret

    if not ret_series:
        log("没有可用的收益率数据，程序退出。")
        return

    raw_all_rets = pd.DataFrame(ret_series)

    if raw_all_rets.empty:
        log("合并后无交易日数据，程序退出。")
        return

    if start_date is not None:
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        raw_all_rets = raw_all_rets[raw_all_rets.index >= start_dt]
    if end_date is not None:
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        raw_all_rets = raw_all_rets[raw_all_rets.index <= end_dt]

    if raw_all_rets.empty:
        log("在指定的日期范围内无数据，程序退出。")
        return

    clean_all_rets = raw_all_rets.dropna().copy()
    if not clean_all_rets.empty:
        clean_all_rets['year_month'] = clean_all_rets.index.to_period('M')

    months = raw_all_rets.index.to_period('M').unique().sort_values()
    total_months = len(months)
    na_months = []
    available_codes = list(ret_series.keys())

    log("月度协方差矩阵")
    for month in months:
        mask_raw = raw_all_rets.index.to_period('M') == month
        raw_month = raw_all_rets[mask_raw]
        org_days = len(raw_month)

        if not clean_all_rets.empty:
            mask_clean = clean_all_rets['year_month'] == month
            month_data = clean_all_rets[mask_clean]
            valid_days = len(month_data)
        else:
            month_data = pd.DataFrame()
            valid_days = 0

        log(f"\n月份: {month}")
        log(f"原数据天数 (并集): {org_days}")
        log(f"有效天数 (交集): {valid_days}")

        missing_counts = raw_month.isna().sum()
        missing_assets = [
            (code, missing_counts.get(code, 0))
            for code in available_codes
            if missing_counts.get(code, 0) > 0
        ]
        if missing_assets:
            log("缺失资产:")
            for code, miss in missing_assets:
                short_name = name_map.get(code, code)
                log(f"  {short_name} ({code}): 缺失 {miss} 天")
        else:
            log("缺失资产: 无")

        if not (valid_days > 1 and not month_data.empty):
            na_months.append(str(month))

    log("\n汇总信息")
    log(f"总月份数: {total_months}")
    log(f"协方差矩阵有效的月份数: {total_months - len(na_months)}")
    log(f"协方差矩阵为NA的月份数: {len(na_months)}")
    if na_months:
        log(f"NA的月份: {', '.join(na_months)}")
    log("")

    if output_txt is not None:
        try:
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_buffer))
            print(f"\n保存至文件: {output_txt}")
        except Exception as e:
            print(f"写入文件失败: {e}")

def extract_cov_sequence(txt_file,
                         start_date=None,
                         end_date=None,
                         h5_file='stock_daily_data.h5',
                         W=1):

    with open(txt_file, 'r') as f:
        codes = [line.strip() for line in f if line.strip()]
    if not codes:
        raise ValueError("资产列表为空，请检查txt文件。")

    # 从 h5 文件读取资产简写
    try:
        name_map_df = pd.read_hdf(h5_file, key='meta/name_map')
        name_map = dict(zip(name_map_df['ts_code'], name_map_df['cnspell']))
    except (KeyError, Exception):
        name_map = {code: code for code in codes}

    ret_series = {}
    for code in codes:
        key = f'data/{code.replace(".", "_")}'
        try:
            df = pd.read_hdf(h5_file, key=key)
        except KeyError:
            continue
        except Exception:
            continue

        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.sort_values('trade_date')
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df_ret = df[['trade_date', 'log_ret']].dropna()
        df_ret = df_ret.set_index('trade_date')['log_ret']
        df_ret.name = code
        ret_series[code] = df_ret

    if not ret_series:
        raise RuntimeError("没有可用的收益率数据，无法生成协方差序列。")

    valid_codes = [code for code in codes if code in ret_series]
    raw_all_rets = pd.DataFrame({code: ret_series[code] for code in valid_codes})

    if raw_all_rets.empty:
        raise RuntimeError("合并后无交易日数据。")

    if start_date is not None:
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        raw_all_rets = raw_all_rets[raw_all_rets.index >= start_dt]
    if end_date is not None:
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        raw_all_rets = raw_all_rets[raw_all_rets.index <= end_dt]

    if raw_all_rets.empty:
        raise RuntimeError("在指定的日期范围内无数据。")

    clean_all_rets = raw_all_rets.dropna().copy()
    if not clean_all_rets.empty:
        clean_all_rets['year_month'] = clean_all_rets.index.to_period('M')

    months = raw_all_rets.index.to_period('M').unique().sort_values()
    name_list = [name_map.get(code, code) for code in valid_codes]
    n_assets = len(valid_codes)
    nan_mat = np.full((n_assets, n_assets), np.nan)

    if W > len(months):
        raise ValueError(f"窗口宽度 W={W} 不能大于总月份数 {len(months)}")

    date_list = [str(m) for m in months[W-1:]]
    S_list = []

    N = len(months)
    for idx in range(W - 1, N):
        window_months = months[idx - W + 1 : idx + 1]

        if not clean_all_rets.empty:
            mask = clean_all_rets['year_month'].isin(window_months)
            window_data = clean_all_rets[mask].drop(columns='year_month')
        else:
            window_data = pd.DataFrame()

        if len(window_data) > 1:
            cov = window_data.cov()
            S_list.append(cov.values)
        else:
            S_list.append(nan_mat.copy())

    S_array = np.stack(S_list, axis=0)   # shape: (K, p, p)

    return name_list, S_array, date_list


def run_network_analysis(portfolio, h5_file, W,
                         type='SGL', gamma=0.1,
                         start_date=None, end_date=None,
                         p_name='portfolio',
                         output_dir='.'):
    """
    从 h5 文件读取数据，提取协方差序列并转为相关系数矩阵，
    运行 FGL/SGL 网络估计，结果保存为 {p_name}_W={W}.pkl。
    """
    N = 20 * W

    name, S, date = extract_cov_sequence(
        txt_file=portfolio, h5_file=h5_file,
        start_date=start_date, end_date=end_date, W=W
    )

    Corr = cov_to_corr(S)

    if type == 'FGL':
        _, _, Theta, Adj_seq = select_best_fgl(Corr, N, gamma=gamma)
    else:
        if type != 'SGL':
            print(f"未知 type='{type}'，默认使用 SGL")
        _, Theta, Adj_seq = select_best_sgl(Corr, N, gamma=gamma)

    l1_pen = compute_l1_distance(Theta)
    jaccard_seq = jaccard_index_sequence(Adj_seq)

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

    filename = os.path.join(output_dir, f"{type}_{p_name}_W={W}.pkl")
    os.makedirs(output_dir, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump({'name': name, 'data_array': data_array}, f)

    print(f"已保存至: {filename}")
    return name, data_array
