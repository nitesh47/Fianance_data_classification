#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:15:20 2019

@author: nitesh
"""
from svm_random_feature import task
from Lstm_feature import ANN
from keras.models import load_model
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
''' For saving the model install (sudo pip install h5py)'''

class Test(object):
    
    def __init__(self,data):
        self.data = data
        self.task = task(data)
        self.lstm = ANN()
        self.model = self.get_model()
        
    def call_clean_text(self, text):
        cleaned_text = self.task.clean_text(text)
        return cleaned_text
    
    def target(self,labels):
        y = self.lstm.label(labels)
        return y        
    
    def lstm_feature(self,text):
        feature_set = self.lstm.feature(text)
        return feature_set
    
    def call_lsmt_build_classifier(self, X):
        classifier = self.lstm.build_classifier(X)
        return classifier
    
    def get_model(self):
        """Get the model object for this instance, loading it if it's not already loaded."""
        model = load_model('model.h5')
        return model
    
if __name__ == '__main__':
     
    
    test_data = input('Enter String: ')
    df =  pd.DataFrame([x.split(';') for x in test_data.split('\n')], columns = ['Zweck'])
        
    test = Test(test_data)
    df['Zweck'] = df['Zweck'].apply(test.call_clean_text)
    
    lstm_feature = test.lstm_feature(df)
    
    model = test.get_model()
    
    prediction = model.predict(lstm_feature)
    
    labels = ['Antidiskriminierung','Arbeit','Bauen Wohnen',
              'Bildung','Bürgerschaftliches Engagement Bürgerbeteiligung','Denkmalschutz',
              'Europa','Familie','Finanzen','Forschung','Frauen',
              'Gesundheit','Gleichstellung','Integration','Jugend','Justiz',
              'Kirchen Religions Weltanschauungsgemeinschaften',
              'Kultur','Pflege','Sicherheit Ordnung','Soziales','Sport',
              'Stadtentwicklung','Umwelt','Verbraucherschutz','Verkehr','Wirtschaft','Wissenschaft']
    print(test_data,labels[np.argmax(prediction)])
    
    
    
