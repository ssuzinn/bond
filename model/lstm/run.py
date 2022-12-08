import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping,
                                        ModelCheckpoint)
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.optimizers import RMSprop
from tqdm import tqdm

#from bond.model.dataset import get

result_path="bond/data/result/lstm_n/"

def run(i):
    bondcodes = os.listdir('bond/data/train_test/'+str(i))
    for bondcode in bondcodes:
        if os.path.isfile(result_path+str(bondcode)+'/'+str(i)+'_checkpoint.h5'):
            pass
        else:
            train_data = np.load('bond/data/train_test/'+str(i)+'/'+str(bondcode)+'/train'+str(i)+'.npz')
            trainX =train_data['x']
            trainY=train_data['y']
            trainLY=train_data['l']
            
            test_data = np.load('bond/data/train_test/'+str(i)+'/'+str(bondcode)+'/test'+str(i)+'.npz')
            testX =test_data['x']
            testY=test_data['y']
            testLY=test_data['l']
            
            dates=np.load('bond/data/train_test/'+str(i)+'/'+str(bondcode)+'/date'+str(i)+'.npy',allow_pickle=True)
                
            enc = OneHotEncoder(sparse=False)
            train_e_y = enc.fit_transform(trainLY.reshape(trainLY.shape[0],trainLY.shape[1]))
            test_e_y = enc.transform(testLY.reshape(testLY.shape[0],testLY.shape[1]))
            
            #lstm model
            def create_model():
                return tf.keras.Sequential([LSTM(25,input_shape=(240,1), recurrent_dropout=0.1),
                                        Dense(2,activation='softmax')])
        
            #with strategy.scope():
            model = create_model()
            model.compile(loss='binary_crossentropy',optimizer=RMSprop(),
                        metrics=['accuracy'])
            earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',restore_best_weights=True)

            filename = result_path+str(bondcode)+'/'+str(i)+'_checkpoint.h5'
            checkpoint = ModelCheckpoint(filename,             # file명을 지정합니다
                                        monitor='val_loss',   # val_loss 값이 개선되었을때 호출됩니다
                                        verbose=1,            # 로그를 출력합니다
                                        save_best_only=True,  # 가장 best 값만 저장합니다
                                        mode='auto'           # auto는 알아서 best를 찾습니다. min/max
                                        )
            model.fit(trainX,train_e_y
                    ,validation_split=0.2, 
                    epochs=1000, batch_size=512,
                    callbacks=[earlystopping,checkpoint])
            
            loss, acc = model.evaluate(testX, test_e_y, verbose=0)
            y_pred = model.predict(testX)
            prob=y_pred[:,1].reshape((1, len(y_pred[:,1]))).tolist()[0]
            y_p=[1 if p > 0.5 else 0 for p in prob]
            
            print('##### Test Result #####')
            print('loss : ',loss)
            print('Accuracy : ',acc)
            
            output_data = pd.DataFrame({'y_pred':y_p, 'y_true': testLY.flatten(), 'y_prob_1': prob,'bondcode': [bondcode]*len(testY.flatten()),
                                        'Date': dates.flatten(), 'return': testY.flatten(),'loss':[loss]*len(testY.flatten()),'accuracy':[acc]*len(testY.flatten())})
            
            os.mkdir(result_path+str(bondcode))
            output_path =result_path+str(bondcode)+'/'+str(i)+'_predict.csv'
            try:
                with open(output_path, "w") as output:
                    output.write(output_data)
            except:
                output_data.to_csv(output_path)

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

def run3():
    for m in tqdm(range(6)):
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
  
            enc = OneHotEncoder(sparse=False)
            train_e_y = enc.fit_transform(trainy.reshape(trainy.shape[0],trainy.shape[1]))
            test_e_y = enc.transform(testy.reshape(testy.shape[0],testy.shape[1]))
            
            #lstm model
            def create_model():
                return tf.keras.Sequential([LSTM(25,input_shape=(120,1), recurrent_dropout=0.1),
                                        Dense(2,activation='softmax')])
        
            #with strategy.scope():
            model = create_model()
            model.compile(loss='binary_crossentropy',optimizer=RMSprop(),
                        metrics=['accuracy'])
            earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',restore_best_weights=True)

            filename = result_path+str(m)+'/'+'checkpoint.h5'
            checkpoint = ModelCheckpoint(filename,             # file명을 지정합니다
                                        monitor='val_loss',   # val_loss 값이 개선되었을때 호출됩니다
                                        verbose=1,            # 로그를 출력합니다
                                        save_best_only=True,  # 가장 best 값만 저장합니다
                                        mode='auto'           # auto는 알아서 best를 찾습니다. min/max
                                        )
            model.fit(trainx,train_e_y
                    ,validation_split=0.2, 
                    epochs=1000, batch_size=512,
                    callbacks=[earlystopping,checkpoint])
            
            loss, acc = model.evaluate(testx, test_e_y, verbose=0)
            y_pred = model.predict(testx)
            prob=y_pred[:,1].reshape((1, len(y_pred[:,1]))).tolist()[0]
            y_p=[1 if p > 0.5 else 0 for p in prob]
            
            print('##### Test Result #####')
            print('loss : ',loss)
            print('Accuracy : ',acc)
            
            bondcodes=np.concatenate([[j[0]]*i.shape[0] for i,j in zip(test_target_date,test_target_bondcode)]).ravel()
                
            output_data = pd.DataFrame({'y_pred': y_pred.flatten(), 
                                        'y_true': testy.ravel(), 
                                        'bondcode': bondcodes,
                                        'Date': np.concatenate(test_target_date).ravel() ,
                                        'return': np.concatenate(testY).ravel(),
                                        'accuracy':[acc]*len(bondcodes),
                                        'y_prob':y_p})
            
            with open(output_path, "w") as output:
                output_data.to_csv(output)


if __name__=='__main__':
    run3()