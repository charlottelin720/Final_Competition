#!/usr/bin/env python
# coding: utf-8

# In[371]:


import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import jieba
from keras.utils.np_utils import *
from keras import layers
#LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from sklearn.utils import shuffle


# In[372]:


# read txt files as sentences
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')


# In[373]:


train_df = shuffle(train_df)
# test_df = shuffle(test_df)


# In[374]:


train_df


# In[375]:


print(train_df.shape)
print(test_df.shape)


# In[376]:


train_title = list(train_df['title'])
train_keyword = list(train_df['keyword'])
test_title = list(test_df['title'])
test_keyword = list(test_df['keyword'])


# In[377]:


train_title_keyword = []
test_title_keyword = []
for x,y in zip(train_title,train_keyword):
    train_title_keyword.append(str(x)+str(y))
for x,y in zip(test_title,test_keyword):
    test_title_keyword.append(str(x)+str(y))


# In[378]:


train_title_keyword[0]


# In[379]:


# title_keyword[1]


# In[380]:


train_df['title_keyword'] = train_title_keyword
test_df['title_keyword'] = test_title_keyword


# In[381]:


train_df['title_keyword(token)'] = train_df.title_keyword.apply(jieba.lcut)
test_df['title_keyword(token)'] = test_df.title_keyword.apply(jieba.lcut)


# In[382]:


# train_df


# In[383]:


train_token = list(train_df['title_keyword(token)'])
test_token = list(test_df['title_keyword(token)'])


# In[384]:


# Read chinese stopword file

all_doc = []
f = open('cn_stopwords.txt','r',encoding="utf-8")
doc = f.read().splitlines()
all_doc.append(doc)
stop_word_all = all_doc[0]


# In[385]:


train_token_no_stopword = []
test_token_no_stopword = []

stop_word = ['。','？','nan','!','，',',','“','”','.','..','...','!','!!','!!!','2','3','4','5','6','7','8','9',':',"'",'s',
                ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '！', '*', '@', '#', '%','：']

for x in train_token:
    temp = []
    for y in x:    
        if y not in stop_word and y not in stop_word_all:
            temp.append(y)
    train_token_no_stopword.append(temp)

for x in test_token:
    temp = []
    for y in x:
        
        if y not in stop_word and y not in stop_word_all:
            temp.append(y)
    test_token_no_stopword.append(temp)


# In[386]:


# test_token_no_stopword[0:5]
# train_token[1]


# In[387]:


from keras.preprocessing.text import Tokenizer
token = Tokenizer()
#reference: https://www.analyticsvidhya.com/blog/2020/03/pretrained-word-embeddings-nlp/

token.fit_on_texts(train_token_no_stopword)


# In[388]:


x_train_seq = token.texts_to_sequences(train_token_no_stopword)
x_test_seq = token.texts_to_sequences(test_token_no_stopword)


# In[389]:


mean = 0
for x in x_test_seq:
    mean += len(x)
print(mean/len(x_test_seq))


# In[390]:


# for x in x_train_seq:
#     print(len(x))


# In[391]:


len(x_test_seq)


# In[392]:


from keras.preprocessing import sequence

#其實也可以使用貼文的平均長度，但會變成只有24個字，效果不是很好，因此使用最常用的長度100

max_review_length = 17
x_train = sequence.pad_sequences(x_train_seq, maxlen = max_review_length)
x_test = sequence.pad_sequences(x_test_seq, maxlen = max_review_length)


# In[393]:


len(x_train)


# In[394]:


x_train[5]


# In[395]:


size_of_vocabulary=len(token.word_index)+1  #+1 for padding
print(size_of_vocabulary)


# In[396]:


# train_df


# In[397]:


y_train = to_categorical(list(train_df.label))


# In[398]:


y_train.shape


# In[399]:


train_df.label


# In[400]:


b[0]


# In[401]:


vocab_size = size_of_vocabulary
embedding_dim = 50


model=Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
      output_dim=embedding_dim,
      input_length=max_review_length))
model.add(layers.LSTM(units=50,return_sequences=True))
model.add(layers.LSTM(units=10))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8))
model.add(layers.Dense(10, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", 
     metrics=['accuracy'])
model.summary()

# #設計: 模型各層定義
# model = Sequential()
# model.add(Embedding(size_of_vocabulary, 300, input_length=max_review_length))
# model.add(LSTM(units=128,return_sequences=True))
# model.add(LSTM(50,activation = 'relu',return_sequences=True))
# model.add(LSTM(units=10,return_sequences=True))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation="sigmoid"))
# model.add(Dense(10,activation='softmax'))
        
# print(model.summary())
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[402]:


#進行訓練
history = model.fit(x_train, y_train, validation_split = 0.2, epochs = 10, batch_size = 64, verbose = 1)

# save the model to disk
import pickle

filename = 'LSTM_model_vr2.sav'
pickle.dump(model, open(filename, 'wb'))


# In[403]:


predict = model.predict_classes(x_test).astype('int')


# In[404]:


np.save('pred_test.npy',predict)


# In[405]:


# pred = np.load('pred_test.npy', allow_pickle=True)
# pred = pred.tolist()
# print(len(pred))


# In[406]:


submit_file = pd.read_csv('sample.csv')
# rf_res = rf.predict(dataTest)
submit_file['label'] = predict
submit_file.to_csv('0853420_ver2.csv', index= False)


# In[407]:


# check = pd.read_csv('0853420.csv')
# print(check, check.shape)
# print(predict, len(predict))


# In[ ]:




