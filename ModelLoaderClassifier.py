# -*- coding: utf-8 -*-
"""
    @author: pkishore
"""
import torch
from torchtext import data
import torch.nn as nn
from torch import optim
import numpy as np
import time, random
import os
from tqdm import tqdm
from gensim.models import KeyedVectors
import torch.nn.functional as F
from torch.autograd import Variable
from BiLSTM import BiLSTM
import csv

test_directory = './data/' #Directory File Path
test_file = 'test-inputs.txt' #Input File name for Classification
test_output_file = 'test-outputs.txt' #Output File name after Classification
EPOCHS = 10
USE_GPU = torch.cuda.is_available()
EMBEDDING_DIM = 300
HIDDEN_DIM = 200
BATCH_SIZE = 50

text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)

train, valid, test = data.TabularDataset.splits(path='./data/dataset', train = 'newtrain.txt',
                                                validation = 'newvalidation.txt',
                                                test = 'newtest.txt',
                                                fields = [('sentence', text_field), ('isQuestion', label_field)],
                                                format = 'csv',
                                                csv_reader_params = {'delimiter': '|'})

text_field.build_vocab(train, valid, test)
label_field.build_vocab(train, valid, test)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using: " + str(USE_GPU))

model = BiLSTM(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab), label_size= 3, use_gpu=USE_GPU, batch_size=BATCH_SIZE)

if USE_GPU:
    model = model.cuda()
model.to(device)
model.load_state_dict(torch.load('./data/dataset/best_model.pth'))
model.eval()

input_data= []
out_store_tensor = []
out_list = []

with open(test_directory + test_file, 'r', encoding="utf-8") as f:
    for line in f:
        input_sent = text_field.preprocess(line)
        for i in range(BATCH_SIZE):
           input_data.append([text_field.vocab.stoi[x] for x in input_sent])

for i in range(int(len(input_data)/BATCH_SIZE)):
    input_data_split= input_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    input_data_split = torch.LongTensor(input_data_split)
    input_data_split = input_data_split.cuda()
    model.hidden = model.init_hidden()
    input_data_split = input_data_split.transpose(0,1)
    output = model(input_data_split)
    output = F.softmax(output, 1)
    out_store_tensor.append(torch.argmax(output[0]))

for i in out_store_tensor:
    out_list.append(i.item())

with open(test_directory + test_output_file, 'w', encoding="utf-8") as f:
    for i in out_list:
        f.write("%s\n" % i)
print("Writing File Done")

