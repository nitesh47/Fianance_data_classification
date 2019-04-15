#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:02:11 2019

@author: nitesh
"""

import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

class task(object):
    
    def __init__(self,data):
        self.data = data
        self.de_stopwords = set(stopwords.words('german'))
        self.en_stopwords = set(stopwords.words('english'))
        self.lb_encode = LabelEncoder()
        self.count_vec = CountVectorizer()
        self.tfidf = TfidfTransformer()
    
    ''' Splitting the data into training set'''
    
    def train_data(self, data):
        data = data[['Politikbereich','Zweck','Betrag']]
        data['Zweck'].replace('', np.nan, inplace=True)
        data['Politikbereich'].replace('', np.nan, inplace=True)
        data.dropna(subset=['Zweck'], inplace=True)
        data.dropna(subset=['Politikbereich'], inplace=True)
        return data
    
    ''' Text preprocessing'''
    
    def clean_text(self,text):
        text = BeautifulSoup(text, "lxml").text
        text = text.lower()
        text = re.sub('<br />', '', text)
        text = re.sub('(\n|\r|\t)+', ' ', text)
        text = re.sub('ß', 'ss', text)
        text = re.sub('’', "'", text)
        text = re.sub('[^a-zA-Z0-9? äöü]+', '', text)
        text = re.sub("\d+", " ", text)
        text = re.sub(' +', ' ', text)
        text = text.split()
        text = [w for w in text if w not in self.de_stopwords]
        text = [w for w in text if w not in self.en_stopwords]
        text = [w for w in text if len(w)>1]
        text = ' '.join(text)
        return text
    
    def without_missing_data(self):
        train = task.load_data()
        train['Zweck'] = train['Zweck'].apply(task.clean_text)
        train['Zweck'].replace('', np.nan, inplace=True)
        train.dropna(subset=['Zweck'], inplace=True)
        return train
    
    '''Encoding categorical data'''
    
    def label_encoder(self, labels):
        
        y = self.lb_encode.fit_transform(labels)
        #print(self.lb_encode.classes_)
        return y
    
    '''Inverse transforming categorical data'''
    
    def inverse_transform_label(self,y_pred):
        y_pred = self.lb_encode.inverse_transform(y_pred)
        return y_pred
    
    def return_class_name(self):
        labels = self.lb_encode.classes_
        return labels
    
    '''
        Fit Countervector and Tfidf on the training dataset. '''
        
    def ConterVec_fit_tfidf(self, data):
        vect_data = self.count_vec.fit_transform(data)
        fit_tfidf = self.tfidf.fit_transform(vect_data)
        return fit_tfidf
    
    '''  Using the transform method of the same object on 
    testing data to create feature representation of test data.  '''
    
    def ConterVec_tfidf_transform(self, data):
        vect_data = self.count_vec.transform(data)
        transform_tfidf = self.tfidf.transform(vect_data)
        return transform_tfidf
