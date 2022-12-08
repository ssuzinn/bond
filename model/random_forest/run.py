import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
#from bond.model.dataset import get
import numpy as np

result_path="bond/data/result/rf_n/"

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

def run():
    for m in tqdm(range(5)):
        output_path = result_path+str(m)+'/'+'predict.csv'
        if os.path.isfile(output_path):
            pass
        else:
           # data load
            trainX,trainY,trainLY,testX,testY,testLY,train_target_date,train_target_bondcode,test_target_date,test_target_bondcode = get(m)
            trainx = np.concatenate(trainX)
            trainy = np.concatenate(trainLY).ravel()
            testx = np.concatenate(testX)
            testy = np.concatenate(testLY).ravel()
            
            #Random Forest
            model = RandomForestClassifier(n_estimators=1000, max_depth=20, random_state = 9, n_jobs=-1)
            model.fit(trainx, trainy)
            t_acc= model.score(trainx,trainy)
            
            y_pred = model.predict(testx)
            y_pred_proba = model.predict_proba(testx)
            prb=np.max(y_pred_proba, axis=1)
            
            acc= model.score(testx,testy)
            
            if os.path.isdir(result_path+str(m)) is False:
                os.mkdir(result_path+str(m))
            
            bondcodes=np.concatenate([[j[0]]*i.shape[0] for i,j in zip(test_target_date,test_target_bondcode)]).ravel()
            output_data = pd.DataFrame({'y_pred': y_pred.flatten(),
                                        'y_true': np.concatenate(testLY).ravel(),
                                        'bondcode': bondcodes,
                                        'Date': np.concatenate(test_target_date).ravel() , 
                                        'return': np.concatenate(testY).ravel(),
                                        'accuracy':[acc]*len(bondcodes),
                                        'y_prob':prb})
            
            with open(output_path, "w") as output:
                output_data.to_csv(output)

if __name__=='__main__':
    run()