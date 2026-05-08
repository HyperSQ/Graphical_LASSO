import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime
from GL import (
    select_best_fgl,
    select_best_sgl,
    threshold_to_adjacency,
    jaccard_index_sequence,
    compute_l1_distance,
    NetworkData
)

def analyze_portfolio(txt_file,
                      start_date=None,
                      end_date=None,
                      h5_file='stock_daily_data.h5',
                      token=None,
                      output_txt=None):
    output_buffer = []

    def log(msg):
        print(msg)
        output_buffer.append(msg)

    if token is None:
        token = 'your_tushare_token_here' 
    pro = ts.pro_api(token)

    with open(txt_file, 'r') as f:
        codes = [line.strip() for line in f if line.strip()]
    if not codes:
        log("资产列表为空，请检查txt文件。")
        return

    ts_code_str = ','.join(codes)
    try:
        basic_df = pro.stock_basic(ts_code=ts_code_str, fields='ts_code,cnspell')
        name_map = dict(zip(basic_df['ts_code'], basic_df['cnspell']))
    except Exception as e:
        log(f"获取股票简写失败: {e}")
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
    na_count = 0
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

        log("各资产缺失天数:")
        missing_counts = raw_month.isna().sum()    
        for code in available_codes:
            miss = missing_counts.get(code, 0)          
            short_name = name_map.get(code, code)
            log(f"  {short_name} ({code}): 缺失 {miss} 天")
        # 若某个月 valid_days <= 1，协方差矩阵记为NA
        if valid_days > 1 and not month_data.empty:
            ret_data = month_data.drop(columns='year_month')
            cov_matrix = ret_data.cov()
            log("协方差矩阵:")
            log(cov_matrix.round(8).to_string())
        else:
            na_count += 1
            log("协方差矩阵: NA (有效天数不足或全为缺失)")

    log("\n汇总信息")
    log(f"总月份数: {total_months}")
    log(f"协方差矩阵为NA的月份数: {na_count}\n")

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
                         token=None,
                         factor=1.0,
                         W=1):

    if token is None:
        token = 'your_tushare_token_here'  
    pro = ts.pro_api(token)

    with open(txt_file, 'r') as f:
        codes = [line.strip() for line in f if line.strip()]
    if not codes:
        raise ValueError("资产列表为空，请检查txt文件。")

    ts_code_str = ','.join(codes)
    try:
        basic_df = pro.stock_basic(ts_code=ts_code_str, fields='ts_code,cnspell')
        name_map = dict(zip(basic_df['ts_code'], basic_df['cnspell']))
    except Exception:
        name_map = {code: code for code in codes}   # 失败时用代码本身

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
            S_list.append(cov.values * factor)
        else:
            S_list.append(nan_mat.copy())

    S_array = np.stack(S_list, axis=0)   # shape: (K, p, p)

    return name_list, S_array, date_list
