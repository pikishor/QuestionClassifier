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

BATCH_SIZE = 50 #Number of examples in the batch.
EPOCHS = 10     #Number of Epochs
USE_GPU = torch.cuda.is_available() #Is GPU Available
EMBEDDING_DIM = 300   #Input or Embedding Dimension
HIDDEN_DIM = 200    #Hidden Dimension
vector_file = './data/embeddings.vec' #Path for Fasttext Embeddings
best_dev_acc = 0.0

text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
#For loading training, validation and test corpuses using split method
train, valid, test = data.TabularDataset.splits(path='./data/dataset', train = 'newtrain.txt',
                                                validation = 'newvalidation.txt',
                                                test = 'newtest.txt',
                                                fields = [('sentence', text_field), ('isQuestion', label_field)],
                                                format = 'csv', 
                                                csv_reader_params = {'delimiter': '|'})
#Construct the Vocab object for text and label field from train, validation and test datasets.
text_field.build_vocab(train, valid, test)
label_field.build_vocab(train, valid, test)
#Train, Test and Validation Iterator
train_iter, valid_iter, test_iter = data.BucketIterator.splits((train, valid, test),
                batch_sizes=(BATCH_SIZE, BATCH_SIZE, BATCH_SIZE), sort_key=lambda x: len(x.sentence), repeat=False)

loss_function = nn.NLLLoss() #Loss Function

# Loading FastText Vectors
print('Loading fasttext vectors.')
embed_space = KeyedVectors.load_word2vec_format(vector_file, binary = False)
print('Finished loading fasttext vectors.')

word_to_idx = text_field.vocab.stoi #Syntactic Sugar to get word indices from vocabulary
pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(text_field.vocab), 300))
pretrained_embeddings[0] = 0

# Populating the required embeddings
for key in tqdm(embed_space.vocab.keys()):
    pretrained_embeddings[word_to_idx[key]-1] = embed_space[key]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using: " + str(USE_GPU))


timestamp = str(int(time.time()))
model = BiLSTM(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,\
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE)
model.to(device)
model.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
model.embeddings.weight.requires_grad = False   # Beacuse we don't want to finetune the embeddings weights, and thus excluded from model.parameters()
best_model = model  #Best model initially intiatlized as Model
optimizer = optim.Adam(model.parameters(), lr=1e-3) #Loss Gradient
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

#Calculating Accuracy
def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right/len(truth)

#For training the model
def train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch, device):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    for batch in tqdm(train_iter, desc='Train epoch '+str(epoch+1)):
        sent, label = batch.sentence, batch.isQuestion
        sent, label = sent.to(device), label.to(device)
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent)
        pred_cpu = pred.cpu()
        pred_label = pred_cpu.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)
    return avg_loss, acc

#For evaluating the model
def evaluate(model, data, loss_function, name, device):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for batch in tqdm(data):
        sent, label = batch.sentence, batch.isQuestion
        sent, label = sent.to(device), label.to(device)
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent)
        pred_cpu = pred.cpu()
        pred_label = pred_cpu.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, label)
        avg_loss += loss.item()
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ': loss %.2f acc %.1f' % (avg_loss, acc*100))
    return acc

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for epoch in range(EPOCHS):
    avg_loss, acc = train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch, device)
    tqdm.write('Train: loss %.2f acc %.1f' % (avg_loss, acc*100))
    with torch.no_grad():
        dev_acc = evaluate(model, valid_iter, loss_function, 'Dev', device)
        if dev_acc > best_dev_acc:
            if best_dev_acc > 0:
                os.system('rm '+ out_dir + '/best_model' + '.pth')
            best_dev_acc = dev_acc
            best_model = model
            torch.save(best_model.state_dict(), out_dir + '/best_model' + '.pth')
            # evaluate on test with the best dev performance model
            test_acc = evaluate(best_model, test_iter, loss_function, 'Test', device)

test_acc = evaluate(model, test_iter, loss_function, 'Final Test', device)
print("Test Accuracy: ", test_acc)

