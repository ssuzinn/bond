import pandas as pd
from datetime import datetime

df= pd.read_csv('./corp_bond_return_file0619_onlystock.csv'
        ,low_memory=False,index_col=0)
# rename 
df.rename(columns={'기준일자':'date','종목코드':'code','만기수익률':'ytm','평가가격(종가)':'close','Duration':'duration','Convexity':'convexity',
                    '적용신용등급':'acr','채권내재등급':'bir','고가(거래정보)':'high','저가(거래정보)':'low','채권거래량(거래정보)':'btv','채권그룹':'bond_g',
                    '이자주기':'interest_c','선후순위구분':'priority_c','보증구분':'guarantee_c','옵션부사채정보':'option_bond_info','발행금리':'issuance_r',
                    '발행금액_x':'price_x','발행사코드':'issue_code','종목명':'name','발행사코드_최초':'issue_code_first','발행사명':'pname','보증정보':'guarantee_info', 
                    '발행일':'sdate', '만기일':'edate','이자지급방식':'ipm','발행이율':'rate', '발행금액_y':'price_y', '원금상환방식':'repay_m','CURRENCY_CODE':'currency_code', 
                    '모집방법':'recruit_m', '주식관련사채정보':'stock_related_info', 'daydiff':'day_diff', 'daydiffmod':'day_diff_mod', 'aftercoupon':'after_coupon',
                    'accured_Int':'accured_int','Coupon':'coupon', 'lag_accured_Int':'lag_accured_int', 'Return':'return','Symbol':'symbol',
                    'FiscalYear':'fiscal_year','특수채여부':'s_bonds_ox','FiscalMonth':'fiscal_month','Frequency':'freq','총자산(천원)':'total_assets',
                    '총부채(천원)':'total_debt', '보통주자본금(천원)':'comm_stock','당기순이익(천원)':'net_income','영업활동으로인한현금흐름(천원)':'cash_flow_from_operating'},inplace =True)

df['date']=pd.to_datetime(df['date'])

df['sdate']=df['sdate'].astype('str')
df['sdate']=pd.to_datetime(df['sdate'])
df['edate']=df['sdate'].astype('str')
df['edate']=pd.to_datetime(df['sdate'])

df.to_csv('./corp_bonds.csv')