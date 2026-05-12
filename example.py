from GL_anl import run_network_analysis

#analyze_portfolio(txt_file='macro_30.txt',h5_file='stock_daily_data_1.h5',output_txt='portfolio_analysis_output.txt',
#                  start_date='20160101', end_date='20260501')

#运行前手动回看一下数据，选出非NA范围

W = [1,2,3,6,8]

for w in W:
    run_network_analysis(portfolio='micro_5.txt', h5_file='stock_daily_data_1.h5', type='FGL', W=w,
                        start_date='20170901', end_date='20260501', p_name="output/FGL_M5",gamma=0.1)