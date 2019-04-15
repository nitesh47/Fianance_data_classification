#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:02:11 2019

@author: nitesh
"""
from sklearn.ensemble import RandomForestClassifier

class random_forest(object):
    def __init__(self):
        []
        
    def random_model(self):
        
        random =RandomForestClassifier(n_estimators=60, max_depth=49,
                             random_state=50,criterion='entropy')
        return random
        
