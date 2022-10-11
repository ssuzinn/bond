import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger 
from tensorflow.keras.optimizers import RMSprop

#df= pd.read_stata('../data/data0.csv')
df= pd.read_csv('data/data0.csv')
bondcodes= list(set(df.BondCode))

scaler = MinMaxScaler()

# Training Parameters
batch_size = 512

# Network Parameters
num_input =8  # Return, YearToMaturity, YTM , Bond_Age , Coupon, D_Vkospi, MKTRF , RF, HML, SMB
timesteps = 3 # timesteps
num_classes = 1 # above or below the median

scale_cols=['Return','YearToMaturity',
    'Bond_Age','Coupon','D_Vkospi',
    'MKTRF', 'RF', 'HML', 'SMB']
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.TPUStrategy(resolver)

for i in bondcodes:
    # read the data
    data = df[df.BondCode==i].sort_values('Date').set_index('Date')
    X=data.loc[:,(data.columns != 'Return')&(data.columns != 'BondCode')]
    X = scaler.fit_transform(X)
    Y=data['Return']

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, shuffle=False)
    
    train_X =np.expand_dims(X_train,axis=0)
    test_X = np.expand_dims(X_test,axis=0)
    train_Y =np.expand_dims(Y_train,axis=0)
    test_Y = np.expand_dims(Y_test,axis=0)
    #enc = OneHotEncoder(sparse=False)
    #train_y = enc.fit_transform(training_label)
    #test_y = enc.transform(testing_label)   

    def create_model():
      model = tf.keras.Sequential()
      model.add(LSTM(12,input_shape=(X_train.shape[0],X_train.shape[1]), recurrent_dropout=0.1))
      model.add(Dense(2))
      return model

    #with strategy.scope():
    model = create_model()
    model.compile(loss='mean_squared_error',optimizer=RMSprop(),
                        metrics=['accuracy'])
    earlystopping = EarlyStopping(monitor='val_loss', patience=5, mode='min',restore_best_weights=True)
    
    checkpoint = ModelCheckpoint('tmp_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    model.fit(train_X,train_Y, 
                epochs=200, batch_size=16,
                callbacks=[earlystopping,checkpoint])
    loss, acc = model.evaluate(train_Y, test_Y, verbose=0)
    y_pred_test_lstm = model.predict(test_X)

    print('##### Test Result #####')
    print('loss : ',loss)
    print('Accuracy : ',acc)
    pred=y_pred_test_lstm[:,1].reshape((1, len(y_pred_test_lstm[:,1]))).tolist()[0]
    output_data = pd.DataFrame({'y_prob': pred, 'y_true': Y_test['Return'], 'BondCode': X_test['BondCode'],
                                    'Date': X_test['Date']})
    output_path =  str(i) + '.csv'
    output_data.to_csv(output_path)