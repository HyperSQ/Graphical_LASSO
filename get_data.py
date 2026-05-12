import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
from time import sleep

pro = ts.pro_api('your token here')

target_codes = [
    '601088.SH', '600938.SH', '601899.SH',
    '600309.SH', '600019.SH', '600585.SH', '601668.SH',
    '600031.SH', '300750.SZ', '600760.SH',
    '002594.SZ', '000333.SZ', '002352.SZ', '600900.SH', '603568.SH',
    '600036.SH', '601318.SH', '600048.SH',
    '600519.SH', '600276.SH', '002714.SZ', '603833.SH', '601888.SH', '600754.SH',
    '688981.SH', '688111.SH', '300308.SZ', '002027.SZ'
]

end_date = datetime.now().strftime('%Y%m%d')
start_date = (datetime.now() - timedelta(days=365*20)).strftime('%Y%m%d')

h5_filename = 'stock_daily_data_1.h5'

print(f"数据获取日期范围: {start_date} 至 {end_date}")

# 获取所有资产的简写（cnspell），存入 name_map
ts_code_str = ','.join(target_codes)
try:
    basic_df = pro.stock_basic(ts_code=ts_code_str, fields='ts_code,cnspell')
    name_map = dict(zip(basic_df['ts_code'], basic_df['cnspell']))
    print(f"成功获取 {len(name_map)} 个资产的简写信息。")
except Exception as e:
    print(f"获取资产简写失败: {e}")
    name_map = {code: code for code in target_codes}

with pd.HDFStore(h5_filename, 'w') as store:
    # 保存简写映射
    name_map_df = pd.DataFrame(
        list(name_map.items()), columns=['ts_code', 'cnspell']
    )
    store.put('meta/name_map', name_map_df, format='table')
    print("已保存资产简写至 meta/name_map")

    for i, code in enumerate(target_codes, 1):
        print(f"正在获取第 {i}/{len(target_codes)} 只股票: {code} ...")
        try:
            df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                df = df.sort_values('trade_date', ascending=True)
                key = f'data/{code.replace(".", "_")}'
                store.put(key, df, format='table')
                print(f"成功获取并存储 {len(df)} 条日线数据。")
            else:
                print(f"未能获取到 {code} 的数据，请检查股票代码是否正确或是否有停牌情况。")

        except Exception as e:
            print(f"获取 {code} 时发生异常: {str(e)}")
        sleep(1)

print(f"已保存至文件: {h5_filename}")