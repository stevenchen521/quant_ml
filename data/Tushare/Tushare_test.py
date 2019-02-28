import tushare as ts
ts.set_token('84c167a397881a21b08f6d066848c371638a48a9622344c1ec1ef232')
pro = ts.pro_api()
df = pro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20180718')