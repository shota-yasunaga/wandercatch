import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def splitData(X, y,n_split=10):
    '''
    iterator constructor
    it does Stratified splits for n_split times
    '''
    sss = StratifiedShuffleSplit(n_splits = n_split,test_size = 0.3,random_state = 123)

    for train_index,test_index in sss.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yield X_train, y_train, X_test, y_test
def KFolder(X,y,n_splits=3):
    '''iterator that gives the train and test data for the X,y
       with the n_splits stratified fold
    '''
    kf = StratifiedKFold(n_splits=3,shuffle = True,random_state=123)
    for train_index,test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yield X_train, y_train, X_test, y_test

def readLabels():
    pass