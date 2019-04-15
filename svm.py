#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:02:11 2019

@author: nitesh
"""

from sklearn.linear_model import SGDClassifier

class svn(object):
    def __init__(self):
        []
        
    def svd_model(self):
        
        sgd =SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
        return sgd
        
