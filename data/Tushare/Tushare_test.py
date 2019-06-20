import tushare as ts
from data.Tushare.config import token
ts.set_token(token=token)
pro = ts.pro_api()
# df = pro.get_hist_data(ts_code='sh', start_date='20010101', end_date='20180718')

# df = pro.daily(ts_code='000001.SH', start_date='20010101', end_date='20180718')


### 获取沪股通、深股通成分数据
# df1 = pro.hs_const(hs_type='SH')

### 获取新股上市列表数据
# df = pro.new_share(start_date='20050901', end_date='20181018')

### 获取指数基础信息
# df = pro.index_basic(market='CSI')
# df = pro.index_dailybasic(trade_date='20181018', fields='ts_code,trade_date,turnover_rate,pe')
# 获取指数每日信息
# df = pro.index_daily(ts_code='000001.SH', start_date='20010101', end_date='20190227')
df = pro.index_weight(index_code='000001.SH', start_date='20190131', end_date='20190131')

print(df.head(5))

### shibor
# df = pro.shibor(start_date='20180101', end_date='20181101')


