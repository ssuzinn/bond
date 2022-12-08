import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import nn
import math

from tqdm import tqdm
#from bond.model.dataset import get

class TFModel(nn.Module):
    def __init__(self,iw, ow, d_model, nhead, nlayers, dropout=0.5):
        super(TFModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(1, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
        ) 

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, srcmask):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)
        output = self.linear(output)[:,:,0]
        output = self.linear2(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask

def evaluate(testx,testy,device,model):
    input = torch.tensor(testx).float().to(device)
    output = torch.tensor(testy).float().to(device)
    model.eval()
    length=len(testy)
    for i in range(length):
        src_mask = model.generate_square_subsequent_mask(input.shape[1]).to(device)
        tgt_mask = model.generate_square_subsequent_mask(output.shape[1]).to(device)

        predictions = model(input, output, src_mask, tgt_mask).transpose(0,1)
        predictions = predictions[:, -1:, :]
        output = torch.cat([output, predictions.to(device)], axis=1)
    return torch.squeeze(output, axis=0).detach().cpu().numpy()[1:]

result_path="bond/data/result/transformer_n/"

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
            
            device = torch.device("cuda")
            lr = 1e-4
            iw= 120
            ow=1
            model = TFModel(iw=iw, ow=ow, d_model=512, nhead=8, nlayers=4, dropout=0.1).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            epoch = 500
            model.train()
            progress = tqdm(range(epoch))
            
            for i in progress:
                batchloss = 0.0
                for (inputs, outputs) in zip(trainx,trainy):
                    optimizer.zero_grad()
                    src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
                    result = model(inputs.float().to(device),  src_mask)
                    loss = criterion(result, outputs[:,:,0].float().to(device))
                    loss.backward()
                    optimizer.step()
                    batchloss += loss
                progress.set_description("loss: {:0.6f}".format(batchloss.cpu().item() / len(trainx)))
            
            result=evaluate(testx,testy,device,model)
            
    