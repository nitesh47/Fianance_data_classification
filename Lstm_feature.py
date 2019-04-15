#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 19:39:54 2019

@author: nitesh
"""
import warnings
warnings.filterwarnings('ignore')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
import numpy as np
import pandas as pd

epochs = 10
emb_dim = 128
batch_size = 256

n_most_common_words = 24914
max_len = 130

class ANN(object):
    def __init__(self):
        []
        
    def label(self, text_data):
        target_labels = pd.get_dummies(text_data['Politikbereich']).values
        return target_labels
        
    def feature(self, data):
        tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(data['Zweck'].values)
        sequences = tokenizer.texts_to_sequences(data['Zweck'].values)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        X = pad_sequences(sequences, maxlen=max_len)
        return X
        
    
    def build_classifier(self,X):
        classifier = Sequential()
        classifier.add(Embedding(n_most_common_words, emb_dim, input_length=X.shape[1]))
        classifier.add(SpatialDropout1D(0.7))
        classifier.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
        classifier.add(Dense(28, activation='softmax'))
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return classifier
        

