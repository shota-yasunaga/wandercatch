
import scipy.io
import numpy as np
import sklearn.utils as sku
import sklearn.preprocessing as prep

def data(folder_name):
    '''
    data 
    '''

    # High valence data
    f = open(folder_name+'/X_high.mat','rb')
    X_high = np.array(scipy.io.loadmat(f)['full_list'])
    X_high = X_high.reshape(len(X_high),numFeatures(X_high.shape))
    y_high = np.ones(len(X_high))
    f.close()
    # Low valence data
    f = open(folder_name+'/X_low.mat','rb')
    X_low  = np.array(scipy.io.loadmat(f)['full_list'])
    print(X_low.shape)
    X_low = X_low.reshape(len(X_low),numFeatures(X_low.shape))
    y_low = np.zeros(len(X_low))

    X = np.vstack((X_high,X_low))
    y = np.concatenate((y_high,y_low))
    print(X.shape)
    print(y.shape)

    return X,y

def getFreqValues(file_path):
    '''
    data 
    '''

    f = open(file_path,'rb')
    freqValues = np.array(scipy.io.loadmat(f)['spectra'])
    f.close()
    return freqValues

def numFeatures(shape):
    '''
    Helpter method for data
    given shapee of array, calculate how many features there are. 
    (multiply all of the dimensions except the 1st)
    '''
    accumulator = 1
    for i in range(len(shape)-1):
        accumulator*=shape[i+1]
    return accumulator

