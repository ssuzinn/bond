import os

import h2o
import numpy as np
import pandas as pd
#from dataset import get
from tensorflow import keras

from h2o.estimators import H2ODeepLearningEstimator
from tqdm import tqdm

result_path="bond/data/result/dnn_n/"

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

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=120, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=60, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=10, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    num_classes=2
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def train_model(datanum,x_train,y_train):
    model = make_model(input_shape=x_train.shape[1:])
    keras.utils.plot_model(model, show_shapes=True)
    
    epochs = 400
    batch_size = 32

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            result_path+str(datanum)+'/'+"best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=0.00001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1),
    ]
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )
    return history
        
def h2oM(trainx,trainy,testx,testy):
    h2o.init()
    h2o.remove_all()
    h2o.ls()
    
    #dnn
    dl = H2ODeepLearningEstimator(
                        hidden=[31,10,5], 
                        loss='CrossEntropy', 
                        epochs=400, 
                        activation='MaxoutWithDropout',
                        input_dropout_ratio = 0.1,
                        hidden_dropout_ratios=[0.5,0.5,0.5],
                        stopping_rounds=5,
                        export_weights_and_biases=True,
                        seed=1,
                        l1 = 0.00001)
    
    def change_data_h2o(dataX,dataLY):
        a=pd.DataFrame(dataX)
        b=pd.DataFrame(dataLY)
        new=pd.concat([a,b],axis=1)
        new.columns = [i for i in range(len(new.columns))]
        return new

    new_train=change_data_h2o(trainx,trainy)
    new_test=change_data_h2o(testx,testy)
    
    def get_d(dataframe):
        x_c=list(dataframe.columns)
        y_c=len(dataframe.columns)-1
        x_c.remove(y_c)
        d=h2o.H2OFrame(dataframe)
        return x_c,y_c,d

    t_x_c,t_y_c,h2oTrain=get_d(new_train)
    _,_,h2oTest=get_d(new_test)

    dl.train(x=t_x_c,y=t_y_c, training_frame = h2oTrain)
    performance = dl.model_performance(h2oTest)
    accuracy   = performance.accuracy()
    
    acc = accuracy[0][1]
    y_pred = dl.predict(h2oTest)
    if os.path.isdir(result_path+str(m)) is False:
        os.mkdir(result_path+str(m))

    bondcodes=np.concatenate([[j[0]]*i.shape[0] for i,j in zip(test_target_date,test_target_bondcode)]).ravel()
        
    output_data = pd.DataFrame({'y_pred': y_pred.flatten(), 
                                'y_true': testy.ravel(), 
                                'bondcode': bondcodes,
                                'Date': np.concatenate(test_target_date).ravel() ,
                                'return': np.concatenate(testY).ravel(),
                                'accuracy':[acc]*len(bondcodes)})
    
    with open(output_path, "w") as output:
        output_data.to_csv(output)
    h2o.shutdown()
    
def run():
    for m in tqdm(range(5)):
        output_path = result_path+str(m)+'/'+'predict.csv'
        if os.path.isfile(output_path):
            pass
        else:
            #data load
            trainX,trainY,trainLY,testX,testY,testLY,train_target_date,train_target_bondcode,test_target_date,test_target_bondcode = get(m)
            trainx = np.concatenate(trainX)
            trainy = np.concatenate(trainLY)
            testx = np.concatenate(testX)
            testy = np.concatenate(testLY)
            
            x_train = trainx.reshape((trainx.shape[0], trainx.shape[1], 1))
            x_test = testx.reshape((testx.shape[0], testx.shape[1], 1))
            y_train = trainy.reshape((trainy.shape[0], trainy.shape[1], 1))
            y_test = testy.reshape((testy.shape[0], testy.shape[1], 1))

            if os.path.isfile(result_path+str(m)+'/'+"best_model.h5"):
                model=keras.models.load_model(result_path+str(m)+'/'+"best_model.h5")
            else:
                train_model(m,x_train,y_train)
                model=keras.models.load_model(result_path+str(m)+'/'+"best_model.h5")
            
            test_loss, acc = model.evaluate(x_test, y_test) 
            y_pred_prob = model.predict(testx)
            prob=y_pred_prob[:,1].reshape((1, len(y_pred_prob[:,1]))).tolist()[0]
            y_p=[1 if p > 0.5 else 0 for p in prob]
            
            if os.path.isdir(result_path+str(m)) is False:
                os.mkdir(result_path+str(m))
        
            bondcodes=np.concatenate([[j[0]]*i.shape[0] for i,j in zip(test_target_date,test_target_bondcode)]).ravel()
            output_data = pd.DataFrame({'y_pred': y_p, 
                                        'y_true': testy.ravel(), 
                                        'bondcode': bondcodes,
                                        'Date': np.concatenate(test_target_date).ravel() ,
                                        'return': np.concatenate(testY).ravel(),
                                        'accuracy':[acc]*len(bondcodes),
                                        'prob':prob})
            
            with open(output_path, "w") as output:
                output_data.to_csv(output)

if __name__=='__main__':
    run()