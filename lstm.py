import pandas as pd

df= pd.read_csv('corp_bonds.csv',low_memory=False,index_col=0)

#df.set_index(['date','code','name'])[['Name','return','sdate','edate','total_assets','total_debt','ytm','duration','rate','price_x','price_y']]

duration= df.set_index(['code','name'])[['duration','return']]
corp= df.set_index(['name'])[['total_assets','total_debt','cash_flow_from_operating','net_income','return','illiquid']]
coupon=df.set_index(['code','name'])[['coupon','ytm','rate','return','after_coupon']]
