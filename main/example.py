import io
import os
from contextlib import redirect_stdout
from GL_anl import run_network_analysis

#analyze_portfolio(txt_file='macro_30.txt',h5_file='stock_daily_data_1.h5',output_txt='portfolio_analysis_output.txt',
#                  start_date='20160101', end_date='20260501')

#运行前手动回看一下数据，选出非NA范围

def run_with_log(portfolio, h5_file, type, W, start_date, end_date, p_name, gamma, output_dir):
    """运行网络分析并将超参搜索输出保存到对应的 txt 文件"""
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, f"{type}_{p_name}_W={W}.txt")
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_network_analysis(portfolio=portfolio, h5_file=h5_file, type=type, W=W,
                             start_date=start_date, end_date=end_date, p_name=p_name,
                             gamma=gamma, output_dir=output_dir)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(buf.getvalue())
    print(f"超参日志已保存至: {txt_path}", flush=True)


W = [1,2,3,6,8]

for w in W:
    run_with_log(portfolio='micro_5.txt', h5_file='stock_daily_data_1.h5', type='FGL', W=w,
                 start_date='20170901', end_date='20260501', p_name="M1", gamma=0.1,
                 output_dir='output')
    run_with_log(portfolio='micro_5.txt', h5_file='stock_daily_data_1.h5', type='SGL', W=w,
                 start_date='20170901', end_date='20260501', p_name="M1", gamma=0.1,
                 output_dir='output')
    run_with_log(portfolio='macro_9.txt', h5_file='stock_daily_data_1.h5', type='SGL', W=w,
                 start_date='20180601', end_date='20260501', p_name="M2", gamma=0.1,
                 output_dir='output')
    run_with_log(portfolio='macro_9.txt', h5_file='stock_daily_data_1.h5', type='FGL', W=w,
                 start_date='20180601', end_date='20260501', p_name="M2", gamma=0.1,
                 output_dir='output')

W=[4,6,8,12,24]

for w in W:
    run_with_log(portfolio='macro_30.txt', h5_file='stock_daily_data_1.h5', type='FGL', W=w,
                 start_date='20180601', end_date='20260501', p_name="M3", gamma=0.1,
                 output_dir='output')
    run_with_log(portfolio='macro_30.txt', h5_file='stock_daily_data_1.h5', type='SGL', W=w,
                 start_date='20180601', end_date='20260501', p_name="M3", gamma=0.1,
                 output_dir='output')
