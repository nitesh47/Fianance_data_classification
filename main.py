#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:02:11 2019

@author: nitesh
"""

from svm_random_feature import task
import pandas as pd
from sklearn.metrics import classification_report
from svn import svn
import matplotlib.pyplot as plt
from imblemrn.over_sampling import SMOTE
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from random_forest import random_forest
from Lstm_feature import ANN
import warnings
warnings.filterwarnings('ignore')

class main(object):
    def __init__(self, data):
        self.data = data
        self.task = task(data)
        self.svn = svn()
        self.random_forest = random_forest()
        self.lstm = ANN()
        
    def call_remove_missing_column(self,data):
        corpus = self.task.train_data(data)
        return corpus
    
    def call_clean_text(self, text):
        cleaned_text = self.task.clean_text(text)
        return cleaned_text
        
    def call_label_encoder(self,label):

        labels = self.task.label_encoder(label)
        return labels
    
    def call_return_class_name(self):
        labels = self.task.return_class_name()
        return labels
    
    def call_label(self,label):
        y = self.lstm.label(label)
        return y
        
    def call_ConterVec_fit_tfidf(self,feature):
        feature_set = self.task.ConterVec_fit_tfidf(feature)
        return feature_set
    
    def call_ConterVec_tfidf_transform(self,feature):
        transform_feature = self.task.ConterVec_tfidf_transform(feature)
        return transform_feature
    
    def lstm_feature(self, arg_text):
        feature_set = self.lstm.feature(arg_text)
        return feature_set
    
    def call_lsmt_build_classifier(self, X):
        classifier = self.lstm.build_classifier(X)
        return classifier
    
    def call_svm(self):
        svm = self.svn.svd_model()
        return svm
    
    def call_random_forest(self):
        random = self.random_forest.random_model()
        return random
        
        
        
if __name__ == '__main__':
     
    
    with open('tempData.csv', encoding = 'unicode_escape') as f:
        data = pd.read_csv(f)
        
    A = main(data)
    
    ''' data divided into a training set and prediction set'''
    

    
    text_data = A.call_remove_missing_column(data)
    text_data['Zweck'] = text_data['Zweck'].apply(A.call_clean_text)
    text_data['Zweck'].replace('', np.nan, inplace=True)
    text_data.dropna(subset=['Zweck'], inplace=True)

    ''' plot classes in training set'''
    plt.figure(3,figsize =(40,40))
    fig = plt.figure(figsize=(8,6))
    plt.ylabel('Counts', fontsize=13)
    text_data.groupby('Politikbereich').Betrag.count().plot.bar(ylim=0)
    plt.show()

    ''' Labels Encoding '''
    
    bytag = text_data.groupby('Politikbereich').aggregate(np.count_nonzero)
    tags = bytag[bytag.Zweck >= 6].index
    text_data = text_data[text_data['Politikbereich'].isin(tags)]
    
    
    ''' Labels Encoding '''
    labels = list(text_data['Politikbereich'])
    y = A.call_label_encoder(labels)
        
    ''' Split train and test data in the ratio of 90:10 '''
    X_train, X_test, y_train, y_test = train_test_split(text_data['Zweck'], y, 
                                                    test_size = 0.3, 
                                                    random_state = 10,
                                                    stratify=y)
    
    X_train = A.call_ConterVec_fit_tfidf(X_train)
    X_test = A.call_ConterVec_tfidf_transform(X_test)
    
    smt = SMOTE()
    [X_train, y_train] = smt.fit_resample(X_train,y_train)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    ''' Model building initialization'''
    print('############################ SVM result###############')
    sgd = A.call_svm()
    
    sgd.fit(X_train, y_train)
    
    '''Prediction on test dataset'''
    svm_y_pred = sgd.predict(X_test)
    #print(A.inverse_transform_label(y_pred))
    print('accuracy %s' % accuracy_score(y_test, svm_y_pred))
    print(classification_report(y_test, svm_y_pred))
    
    print(confusion_matrix(y_test, svm_y_pred))
    
    
    print('######################## random forest result ###############')
    
    random = A.call_random_forest()
    
    random.fit(X_train, y_train)
    
    '''Prediction on test dataset'''
    random_y_pred = random.predict(X_test)
    #print(A.inverse_transform_label(y_pred))
    print('accuracy %s' % accuracy_score(y_test,random_y_pred))
    print(classification_report(y_test, random_y_pred))
    print(confusion_matrix(y_test, random_y_pred))
    
    print('######################## LSTM results##############')
          
    target_labels = A.call_label(text_data)
    print('Shape of label tensor:', target_labels.shape)
    
    data = A.lstm_feature(text_data)
    
    ''' Split train and test data in the ratio of 90:10 '''
    train_X, test_X, train_y, test_y = train_test_split(data, target_labels, 
                                                    test_size = 0.3, 
                                                    random_state = 10,
                                                    stratify=target_labels)
    
    
    ''' Upsampling method----> to balance the classes only on training data
    to avoid the information leakage problem'''
    smt = SMOTE() 
    [train_X, train_y] = smt.fit_resample(train_X,train_y)

    classifier = A.call_lsmt_build_classifier(train_X)
    classifier.fit(train_X, train_y, batch_size = 512, epochs = 10)
    
    
    y_pred = classifier.predict(test_X)
    
    classifier.save('model.h5')
    
    print(accuracy_score(np.argmax(test_y,axis=1),
                           np.argmax(y_pred,axis=1)))
    
    print(classification_report(np.argmax(test_y,axis=1),
                           np.argmax(y_pred,axis=1)))



    print(confusion_matrix(np.argmax(test_y,axis=1),
                            np.argmax(y_pred,axis=1)))
