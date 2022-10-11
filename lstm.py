import pandas as pd
import numpy as np
import os

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
df= pd.read_csv('vscode/bond/data/data0.csv',index_col=-0)
bondcodes= list(set(df.BondCode))
df.drop(['Unnamed: 0'], axis = 1, inplace = True)

scaler = MinMaxScaler()

# Training Parameters
batch_size = 512

# Network Parameters
num_input =8  # Return, YearToMaturity, YTM , Bond_Age , Coupon, D_Vkospi, MKTRF , RF, HML, SMB
timesteps = 3 # timesteps
num_classes = 1 # above or below the median

feature_cols=['Return','YearToMaturity',
    'Bond_Age','Coupon','D_Vkospi',
    'MKTRF', 'RF', 'HML', 'SMB']
label_cols=['Return']

# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.TPUStrategy(resolver)

def make_dataset(data, label, window_size=12):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)


for i in bondcodes:
    # read the data
    data = df[df.BondCode==i].sort_values('Date').set_index('Date')
    X=data[feature_cols]
    scale_cols=X.columns
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)
    X.columns = scale_cols
    Y=data[label_cols]

    X, Y = make_dataset(X, Y, 1)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, shuffle=False)

    #enc = OneHotEncoder(sparse=False)
    #train_y = enc.fit_transform(training_label)
    #test_y = enc.transform(testing_label)   

    def create_model():
      model = tf.keras.Sequential()
      model.add(LSTM(6,input_shape=(X_train.shape[1],X_train.shape[2]), recurrent_dropout=0.1))
      model.add(Dense(2))
      return model

    #with strategy.scope():
    model = create_model()
    model.compile(loss='mean_squared_error',optimizer=RMSprop(),
                        metrics=['accuracy'])
    earlystopping = EarlyStopping(monitor='val_loss', patience=5, mode='min',restore_best_weights=True)

    checkpoint = ModelCheckpoint('tmp_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    model.fit(X_train,Y_train, 
                epochs=200, batch_size=12,
                callbacks=[earlystopping,checkpoint])
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    y_pred_test_lstm = model.predict(X_test)

    print('##### Test Result #####')
    print('loss : ',loss)
    print('Accuracy : ',acc)
    pred=y_pred_test_lstm[:,1].reshape((1, len(y_pred_test_lstm[:,1]))).tolist()[0]
    output_data = pd.DataFrame({'y_prob': pred, 'y_true': np.hstack(Y_test) })
    output_path =  str(i) + '.csv'
    output_data.to_csv(output_path)