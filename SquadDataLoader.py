# -*- coding: utf-8 -*-
"""
@author: pkishore
"""
import csv
import json
from tqdm import tqdm
import numpy as np
import os
import nltk
import random
from random import shuffle

data_directory = './data/squad/' #Path to the Data Directory for SQUAD is needed.
test_directory = './data/'
test_file = 'test-inputs.txt'

# min length, max length, avg length, std deviation, total number of questions for each question file
def getStats(filename):
    lengthstore = []
    print(filename)
    filehandle = open(data_directory + filename, 'r')        
    data = json.load(filehandle)
    
    if ('data' in filename):
        for question in tqdm(data['q'], total=len(data['q'])):
            lengthstore.append(len(question))
        filehandle.close()
    else:
        for topic in tqdm(data['p'], total=len(data['p'])):
            sentences = []
            for paragraph in topic:
                for sentence in nltk.sent_tokenize(paragraph):
                    sentences.append(sentence)

            for sentence in sentences:
                lengthstore.append(len(nltk.word_tokenize(sentence)))
    
    print("Minimum Length:", min(lengthstore))
    print("Maximum Length:", max(lengthstore))
    print("Mean Length:", np.mean(lengthstore))
    print("Standard Deviation Length:", np.std(lengthstore))
    print("Number of Items:", len(lengthstore))
    return


datasets = os.listdir(data_directory)           #Lists Directory
#To read and process all the files in SQUAD Dataset: Train, Dev and Test
for file in datasets:
    getStats(file)

lengthstore = [] # for storing length of sentences.
with open(test_directory + test_file, 'r', encoding="utf-8") as f:
    for line in f:
        sentences = nltk.sent_tokenize(line)
        for sentence in sentences:
            lengthstore.append(len(nltk.word_tokenize(sentence)))

print("Minimum Length of Text File:", min(lengthstore))
print("Maximum Length of Text File:", max(lengthstore))
print("Mean Length of Text File:", np.mean(lengthstore))
print("Standard Deviation Length of Text File:", np.std(lengthstore))
print("Number of Lines of Text File:", len(lengthstore))
print("Number of Sets of Lines in Text File:", set(lengthstore))
    

sentencestore = {'questions': [], 'context': []}
# To process SQUAD Dataset into Questions and Context(Not Questions) 
def getSentences(filename, store):
    if ('test' in filename):
        return
    filehandle = open(data_directory + filename, 'r')        
    data = json.load(filehandle)
    
    if ('data' in filename):
        for question in tqdm(data['q'], total=len(data['q'])):
            store['questions'].append(question)
        filehandle.close()
    else:
        for topic in tqdm(data['p'], total=len(data['p'])):
            sentences = []
            for paragraph in topic:
                for sentence in nltk.sent_tokenize(paragraph):
                    sentences.append(sentence)

            for sentence in sentences:
                store['context'].append(nltk.word_tokenize(sentence))
    return

for file in datasets:
    getSentences(file, sentencestore)
    
for key in sentencestore.keys():
    print(key + ':' + str(len(sentencestore[key])))
    
questionlengths = []
contextlengths = []

for question in sentencestore['questions']:
    length = len(question)
    if (length not in questionlengths):
        questionlengths.append(length)
for ctx in sentencestore['context']:
    length = len(ctx)
    if (length not in contextlengths):
        contextlengths.append(length)
print("Question lengths:", questionlengths)
print("Context lengths:", contextlengths)
print("Minimum Context length:", min(contextlengths))
print("Maximum Context length:", max(contextlengths))
print("Minimum Question length:", min(questionlengths))
print("Maximum Question length:", max(questionlengths))

sentencestore['context60'] = []
sentencestore['context270'] = []
for ctx in sentencestore['context']:
    if (len(ctx) <= 60):
        sentencestore['context60'].append(ctx.copy())
    if (len(ctx) <= 270):
        sentencestore['context270'].append(ctx.copy())
        
for key in sentencestore.keys():
    print(key + ':' + str(len(sentencestore[key])))
    
sentencestore['nonquestion'] = []
sentencestore['nonquestion60'] = []
sentencestore['nonquestion270'] = []

random.seed(1234)

for key in sentencestore.keys():
    shuffle(sentencestore[key])
    
for key in sentencestore.keys():
    print(key + ':' + str(len(sentencestore[key])))
    
writefolder = './data/'
foldername = 'dataset'
folders = ['', '60', '270']
datasplit = [['questions', 'nonquestion'], ['questions', 'nonquestion60'], ['questions', 'nonquestion270']]

def writeDataset(store, threshold):
    questiondatastore = []
    contextdatastore = []
    test = []
    train = []
    validation = []
    datasplit = ['questions', 'nonquestion' + threshold]
    for key in datasplit:
        for sentence in tqdm(sentencestore[key], total=len(sentencestore[key])):
            if (key == 'questions'):
                questiondatastore.append({'sentence': ' '.join(sentence), 'isquestion': 1})
            else:
                contextdatastore.append({'sentence': ' '.join(sentence), 'isquestion': 0})
    length = len(questiondatastore)
    ratio = 1/8
    valstart = int(length * ratio)
    test.extend(questiondatastore[:valstart])
    test.extend(contextdatastore[:valstart])
    validation.extend(questiondatastore[valstart:valstart*2])
    validation.extend(contextdatastore[valstart:valstart*2])
    train.extend(questiondatastore[valstart*2:])
    train.extend(contextdatastore[valstart*2:])
    
    shuffle(test)
    shuffle(train)
    shuffle(validation)
    
    with open(writefolder + foldername + threshold + '/test.txt', 'w') as f:
        w = csv.DictWriter(f, fieldnames = ['sentence', 'isquestion'], delimiter = '|')
        w.writeheader()
        for row in tqdm(test, total=len(test)):
            w.writerow(row)
    with open(writefolder + foldername + threshold + '/train.txt', 'w') as f:
        w = csv.DictWriter(f, fieldnames = ['sentence', 'isquestion'], delimiter = '|')
        w.writeheader()
        for row in tqdm(train, total=len(train)):
            w.writerow(row)
    with open(writefolder + foldername + threshold + '/validation.txt', 'w') as f:
        w = csv.DictWriter(f, fieldnames = ['sentence', 'isquestion'], delimiter = '|')
        w.writeheader()
        for row in tqdm(validation, total=len(validation)):
            w.writerow(row)

writeDataset(sentencestore, '')
writeDataset(sentencestore, '60')
writeDataset(sentencestore, '270')

# For removing 50% of questiions marks in questions and 50% of periods in context
def QuestionProcessing(datasetfolder, fileName, newFileName):
    dataset = pd.read_csv(datasetfolder + fileName, delimiter = '|')
    newdataset = dataset.copy()
    qwithmark, qwithoutmark = 0, 0
    for index, row in tqdm(newdataset.iterrows(), total = newdataset.shape[0]):
        if (row['isquestion'] == 1 and '?' == row['sentence'][-1]):
            if (random.random() < 0.5):
                newdataset.loc[index, 'sentence'] = row['sentence'][:-1]
            if ('?' in row['sentence']):
                qwithmark += 1
        elif (row['isquestion'] == 1):
            qwithoutmark += 1
        elif (row['isquestion'] == 0 and '.' == row['sentence'][-1]):
            if (random.random() < 0.5):
                newdataset.loc[index, 'sentence'] = row['sentence'][:-1]
    newdataset.to_csv(datasetfolder + newFileName, index = False, sep = '|')

datasetfolder = './data/dataset/'
fileName = 'train.txt'
newFileName = 'new_train.txt'
QuestionProcessing(datasetfolder, fileName, newFileName)
fileName = 'test.txt'
newFileName = 'new_test.txt'
QuestionProcessing(datasetfolder, fileName, newFileName)
fileName = 'validation.txt'
newFileName = 'new_validation.txt'
QuestionProcessing(datasetfolder, fileName, newFileName)
