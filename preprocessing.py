import pandas as pd

df = pd.read_stata('bond/data/Full_file_0711_empv2.dta')
temp = df[['BondCode','Date','Return']]
bondcodes= list(set(df.BondCode))
L=[]
for i in bondcodes:
    data= temp[temp.BondCode==i].sort_values('Date').set_index('Date')
    if len(data) < 24: #(약 2년)
        pass
    else:
        L.append(i)

print(len(L))
df= temp[temp.BondCode.isin(L)]
df.to_csv('bond/data/bond_return.csv')