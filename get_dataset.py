import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler

def Del(fpath):
    if os.path.exists(fpath):
        for file in os.scandir(fpath):
            os.remove(file.path)
            print('remove all!')
        else:
            print('no files')
        os.rmdir(fpath)

def bond_data_only_return():
    df = pd.read_csv('bond/data/Bond_return_file0712.csv',index_col=0,low_memory=False)
    df.rename(columns = {'종목코드':'BondCode','기준일자':'Date'},inplace=True)
    temp = df[['BondCode','Date','Return']]
    #temp.loc[:,'Return']=winsorize(temp['Return'],limits=[0.01,0.01]) #winsorize 
    temp.to_csv('bond/data/returns.csv')

def get_label(df):
    output = pd.DataFrame()
    df=df.dropna(subset=['Return']) #######!!!!!
    def target_class(return_, median):
        if return_ >= median:
            return 1
        else:
            return 0
    date_list=df.Date.unique()
    for date in tqdm(date_list):
        sub_data = df[df.Date == date]
        return_median = sub_data['Return'].median()
        sub_data['y'] = sub_data['Return'].apply(lambda x: target_class(x,return_median))
        output = pd.concat([output,sub_data], ignore_index = True)
    output.to_csv('bond/data/label_returns.csv')
    
def Normalize_split_study_periods(df):
    def normalize(a,mean,std):
        return (a-mean)/std
    date_list=df.Date.unique()
    date_list=sorted(date_list)
    output=df.sort_values('Date')
    
    
    for i in tqdm(range(22)):
        #1000일씩 split
        # study_date = date_list[1000*i : 1000*(i+1)]        
        # study_period = output[output.Date.isin(study_date)]
        
        # #1000일에서 750일은 train
        # train_date = study_date[:750]
        # train_data = output[output.Date.isin(train_date)]
        
        study_date = date_list[500*i : 500*(i+1)]        
        study_period = output[output.Date.isin(study_date)]
        
        #1000일에서 375 train
        train_date = study_date[:375]
        train_data = output[output.Date.isin(train_date)]
        
        train_price_mean = train_data.Return.mean()
        train_price_std = train_data.Return.std()   
        study_period.loc[:,'Normalized_Price_Return'] = study_period.loc[:,'Return'].apply(normalize,args=(train_price_mean,train_price_std))
        
        if len(study_period) !=0:
            with open('bond/data/train_test/n_normalized/normalized_return_'+str(i)+'.csv','w')as f:
                study_period.to_csv(f)
        else:
            break

#window dataset 생성
def windowDataset(data ,input_window, output_window, stride=1 , binary=None,date=None):
    #총 데이터의 개수
    L = data.shape[0]
    #stride씩 움직일 때 생기는 총 sample의 개수
    num_samples = (L - input_window -output_window) // stride + 1
    #input과 output : shape = (window 크기, sample 개수)
    
    valueX = np.zeros([input_window, num_samples])
    valueY = np.zeros([output_window, num_samples])
    
    if binary is not None:
        labelY = np.zeros([output_window, num_samples])
    
    if date is not None:
        YD=[]
        XD=[]
    
    for i in np.arange(num_samples):
        start_x = stride*i
        end_x = start_x + input_window
        valueX[:,i] = data[start_x:end_x]

        start_y = stride*i + input_window
        end_y = start_y + output_window
        valueY[:,i] = data[start_y:end_y]
        
        if binary is not None:
            start_y = stride*i + input_window
            end_y = start_y + output_window
            labelY[:,i] = binary[start_y:end_y]
        
        if date is not None:
            start_y = stride*i + input_window
            end_y = start_y + output_window
            yd=[date[start_y:end_y]]
            
            start_x = stride*i
            end_x = start_x + input_window
            xd=date[start_x:end_x]
            
            YD.append(yd)
            XD.append(xd)
                    
    valueX = valueX.reshape(valueX.shape[0], valueX.shape[1]).transpose((1,0))
    valueY = valueY.reshape(valueY.shape[0], valueY.shape[1]).transpose((1,0))
   
    if binary is not None:
        labelY = labelY.reshape(labelY.shape[0], labelY.shape[1]).transpose((1,0))
            
    if date is not None:
        xd=np.array(XD)
        Xdates = xd.reshape(xd.shape[0], xd.shape[1])
        yd=np.array(YD)
        Ydates = yd.reshape(yd.shape[0], yd.shape[1])
        
    return valueX,valueY,labelY, Xdates,Ydates

def windowD(data ,input_window, output_window, stride=1,datatype='int'):
    #총 데이터의 개수
    L = data.shape[0]
    #stride씩 움직일 때 생기는 총 sample의 개수
    num_samples = (L - input_window -output_window) // stride + 1
    #input과 output : shape = (window 크기, sample 개수)
    
    if datatype =='str':
        valueX=[]
        valueY=[]
        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            xd= data[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            yd = data[start_y:end_y]
            
            valueX.append(xd)
            valueY.append(yd)
            
        Xd=np.array(valueX)
        valueX = Xd.reshape(Xd.shape[0], Xd.shape[1])
     
        Yd=np.array(valueY)
        valueY = Yd.reshape(Yd.shape[0], Yd.shape[1])
    
    else:
        valueX = np.zeros([input_window, num_samples])
        valueY = np.zeros([output_window, num_samples])
    
        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            valueX[:,i] = data[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            valueY[:,i] = data[start_y:end_y]
              
        valueX = valueX.reshape(valueX.shape[0], valueX.shape[1]).transpose((1,0))
        valueY = valueY.reshape(valueY.shape[0], valueY.shape[1]).transpose((1,0))
        
    return valueX,valueY

# 채권별 데이터셋 모든 기간별 생성 -> 채권별 반환
def run(df,train_test):
    
    def get_Train_Test():
        T=[]
        Te=[]
        for i in range(11):
            df= pd.read_csv('bond/data/train_test/normalized_return_'+str(i)+'.csv',index_col=0)
            df['periods']=[i]*len(df)
            dlist=df.Date.unique()
            tr=df[df.Date.isin(dlist[:750])]
            te=df[df.Date.isin(dlist[510:])]
            
            T.append(tr.reset_index(drop=True))
            Te.append(te.reset_index(drop=True))
        pd.concat(T).reset_index(drop=True).to_csv('bond/data/train_test/Train.csv')
        pd.concat(Te).reset_index(drop=True).to_csv('bond/data/train_test/Test.csv')
    
    get_Train_Test()
    
    if train_test == 'Test':
        Train=pd.read_csv('bond/data/train_test/Train.csv')
        bondCodes = list(Train.BondCode.unique())
    else:
        bondCodes = list(df.BondCode.unique())
    
    for bondcode in tqdm(bondCodes):
        #print(bondcode)
        # read the data
        data=df[df.BondCode == bondcode].dropna(subset=['Return'])
        if len(data) !=0:
            data=data.sort_values('Date')
            #파일존재시 skip
            if os.path.isfile('bond/data/train_test/'+train_test+'/'+str(bondcode)+'/'+'XY.npz'):
                pass
            else:
                if (len(data) >= 241):
                    # 수익률, 바이너리 y , predict 할 날짜
                    return_data = data.loc[:,'Return']
                    binary_y = data.loc[:,'y']
                    dates =data.loc[:,'Date'].tolist()
                    
                    #240일 데이터 보고 다음 날 예측
                    iw = 1*240   
                    ow = 1*1 
                    
                    x =return_data.to_numpy()
                    y = binary_y.to_numpy()
                    
                    #input window, output window, stride를 입력받고 iw만큼의 길이를 stride간격으로 sliding하면서 데이터셋을 생성
                    X,Y,LY,XD,YD = windowDataset(x, input_window=iw, output_window=ow, stride=1,binary=y,date=dates)
                    
                    #폴더 생성 (없을시만)
                    try:
                        os.mkdir('bond/data/train_test/'+train_test+'/'+str(bondcode))
                    except:
                        continue
                    base='bond/data/train_test/'+train_test+'/'+str(bondcode)
                    date_p =base+'/date'
                    p =base+'/XY'
                    
                    np.save(date_p,YD)
                    np.savez(p,x=X,y=Y,l=LY)

#1000일씩 나눈 데이터 기준 채권별 데이터셋 생성 -> 기간별 반환  (Memory error)           
def run2(periods,TT):
    df=pd.read_csv('bond/data/train_test/normalized_return_'+str(periods)+'.csv')
    
    dlist=df.Date.unique()
    if TT =='Train':
        df=df[df.Date.isin(dlist[:750])]
    else:
        df=df[df.Date.isin(dlist[510:])]
    
    datelist= df.Date.unique()
    bondCodes=df.BondCode.unique()
    
    XL=[]
    YL=[]
    LYL=[]
    XDD=[]
    YDD=[]
    
    
    for bondcode in tqdm(bondCodes):
        #print(bondcode)
        # read the data
        data=df[(df.BondCode == bondcode)&(df.Date.isin(datelist))]
        if len(data) !=0:
            data=data.sort_values('Date')
            #파일존재시 skip
            if os.path.isfile('bond/data/train_test/'+str(periods)+'_'+str(TT)+'.csv'):
                pass
            else:
                if (len(data) >= 241):
                    # 수익률, 바이너리 y , predict 할 날짜
                    return_data = data.loc[:,'Return']
                    binary_y = data.loc[:,'y']
                    dates =data.loc[:,'Date'].tolist()
                    
                    #240일 데이터 보고 다음 날 예측
                    iw = 1*240   
                    ow = 1*1 
                    
                    x =return_data.to_numpy()
                    y = binary_y.to_numpy()
                    
                    #input window, output window, stride를 입력받고 iw만큼의 길이를 stride간격으로 sliding하면서 데이터셋을 생성
                    X,Y,LY,XD,YD = windowDataset(x, input_window=iw, output_window=ow, stride=1,binary=y,date=dates)
                    XL.append(X)
                    YL.append(Y)
                    LYL.append(LY)
                    XDD.append(XD)
                    YDD.append(YD)
                

    DF=[]  
    base='bond/data/train_test/'+str(periods)+'_'+str(TT)+'.pkl'
    for x,y,ly,xd,yd in zip(XL,YL,LYL,XDD,YDD):

        for i in range(len(x)):
            a=pd.DataFrame(x[i],columns=['Return'])
            d=pd.DataFrame(xd[i],columns=['Date'])
            A=pd.concat([a,d],axis=1)
            
            b=pd.DataFrame(y[i],columns=['Return'])
            c=pd.DataFrame(ly[i],columns=['R_binary'])
            e=pd.DataFrame(yd[i],columns=['Date'])
            B=pd.concat([b,c,e],axis=1)
            ADF=pd.concat([A,B],axis=0)
            DF.append(ADF)
    pd.concat(DF).to_pickle(base)

#기간별 데이터 반환
def run3(TT):
    base='bond/data/train_test/'
    for m in range(6):
        if os.path.isfile(base+TT+'/'+str(m)+'/'+'XY.npz'):
                pass
        else:
            data = pd.read_csv(base+'n_normalized/'+'normalized_return_'+str(m)+'.csv',index_col = 0)
            data = data.sort_values(['Date','BondCode'],ascending = True)
            date_list = list(data.Date.unique())
            bond_list = list(data.BondCode.unique())
            bond_list.sort()
          
            # Generate Training Set
            if TT=='n_Train':
                sub_date_list = date_list[0 : 375]
            else:
                sub_date_list = date_list[255 :]
            
            XL=[]
            YL=[]
            LYL=[]
            XDD=[]
            YDD=[]
            B=[]

            for bond in tqdm(bond_list):
                
                bond_data = data[data.BondCode == bond]
                bond_data = bond_data[bond_data.Date.isin(sub_date_list)]
                if len(bond_data) >= 121:
                    dates =bond_data.loc[:,'Date']
                
                    return_data=bond_data.loc[:,'Normalized_Price_Return']
                    binary_y=bond_data.loc[:,'y']
                    
                    #120 데이터 보고 다음 날 예측
                    iw = 1*120   
                    ow = 1*1 
                    
                    x =return_data.to_numpy()
                    y = binary_y.to_numpy()
                    
                    #input window, output window, stride를 입력받고 iw만큼의 길이를 stride간격으로 sliding하면서 데이터셋을 생성
                    X,Y= windowD(x, input_window=iw, output_window=ow, stride=1)
                    LX,LY= windowD(y, input_window=iw, output_window=ow, stride=1)
                    XD,YD= windowD(dates, input_window=iw, output_window=ow, stride=1,datatype='str')
                    
                    XL.append(X)
                    YL.append(Y)
                    LYL.append(LY)
                    XDD.append(XD)
                    YDD.append(YD)
                    B.append([bond])
                
            #폴더 생성 (없을시만)
            try:
                os.mkdir('bond/data/train_test/'+TT+'/'+str(m))
            except:
                continue
            baseP='bond/data/train_test/'+TT+'/'+str(m)
            date_p =baseP+'/target_date_bondcode'
            p =baseP+'/XY'
            
            np.savez(date_p,targetdate=np.array(YDD),bondcode=np.array(B))
            np.savez(p,x=np.array(XL),y=np.array(YL),l=np.array(LYL))
                    
if __name__=='__main__':
    #bond_data_only_return()
    print('1. done!')
    #r_df= pd.read_csv('bond/data/returns.csv',index_col=-0)
    #get_label(r_df)
    print('2. done!')
    #l_df= pd.read_csv('bond/data/label_returns.csv',index_col=-0)
    #Normalize_split_study_periods(l_df)
    print('3. done!')
    run3("n_Train")
    run3("n_Test")
    print('!!!')