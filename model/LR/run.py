import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from tqdm import tqdm

#from bond.model.dataset import get

result_path="bond/data/result/lr_n/"

def get(m):
    train_base_root= 'bond/data/train_test/n_Train/'
    test_base_root= 'bond/data/train_test/n_Test/'

    train_data = np.load(train_base_root+str(m)+'/XY.npz',allow_pickle=True)
    trainX = train_data['x']
    trainY= train_data['y']
    trainLY= train_data['l']

    test_data = np.load(test_base_root+str(m)+'/XY.npz',allow_pickle=True)
    testX =test_data['x']
    testY=test_data['y']
    testLY=test_data['l']

    Train_dates_bondcode=np.load(train_base_root+str(m)+'/target_date_bondcode.npz',allow_pickle=True)
    train_target_date = Train_dates_bondcode['targetdate']
    train_target_bondcode = Train_dates_bondcode['bondcode']

    Test_dates_bondcode=np.load(test_base_root+str(m)+'/target_date_bondcode.npz',allow_pickle=True)
    test_target_date = Test_dates_bondcode['targetdate']
    test_target_bondcode = Test_dates_bondcode['bondcode']
    
    return trainX,trainY,trainLY,testX,testY,testLY,train_target_date,train_target_bondcode,test_target_date,test_target_bondcode

def logisticRegression(trainx,trainy,testx,testy):
    #logistic regression
    model = LogisticRegressionCV(cv=2,penalty='l2',solver='lbfgs',verbose=1)   
    model.fit(trainx,trainy)
    y_pred = model.predict(testx)
    acc=model.score(testx,testy)
    prb = np.max(model.predict_proba(testx), axis=1)
    
    return y_pred,acc,prb

def save_output(m,test_target_date,test_target_bondcode,y_pred,testy,testY,acc,prb,output_path):
    if os.path.isdir(result_path+str(m)) is False:
                    os.mkdir(result_path+str(m))
                
    bondcodes=np.concatenate([[j[0]]*i.shape[0] for i,j in zip(test_target_date,test_target_bondcode)]).ravel()
    output_data = pd.DataFrame({'y_pred': y_pred.flatten(), 
                                'y_true': testy.ravel(), 
                                'bondcode': bondcodes,
                                'Date': np.concatenate(test_target_date).ravel() ,
                                'return': np.concatenate(testY).ravel(),
                                'accuracy':[acc]*len(bondcodes),
                                'y_prob':prb})
    
    with open(output_path, "w") as output:
        output_data.to_csv(output)
        
def run():
    for m in tqdm(range(5)):
        output_path = result_path+str(m)+'/'+'predict.csv'
        if os.path.isfile(output_path):
            pass
        else:
            # data load
            trainX,trainY,trainLY,testX,testY,testLY,train_target_date,train_target_bondcode,test_target_date,test_target_bondcode = get(m)
            trainx = np.concatenate(trainX)
            trainy = np.concatenate(trainLY)
            testx = np.concatenate(testX)
            testy = np.concatenate(testLY)
            
            y_pred,acc,prb = logisticRegression(trainx,trainy,testx,testy)
            
            save_output(m,test_target_date,test_target_bondcode,y_pred,testy,testY,acc,prb,output_path)
                
if __name__=='__main__':
    run()