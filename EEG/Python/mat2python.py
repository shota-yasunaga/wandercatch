
import os
import scipy.io as io
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

def getFreqValuesVec(file_path):
    '''
    data 
    '''

    f = open(file_path,'rb')
    freqValues = np.array(io.loadmat(f)['spectra'])
    freqVec   = np.array(io.loadmat(f)['freqVec'])
    f.close()
    return freqValues,freqVec


def getChanLocs(file_path):
    f = open(file_path,'rb')
    mat_labels = io.loadmat(f)['labels'][0]
    f.close()
    labels = ['']*len(mat_labels)
    for i in range(len(mat_labels)):
        labels[i] = mat_labels[i][0]
    return labels

    

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


def folderIterator(folder_path):
    folder_dir = os.fsencode(folder_path)
    for file in os.listdir(folder_dir):
        folder_full_path = os.path.join(folder_dir,file)
        yield folder_full_path

def chanIndIterator(loc_path,chan='Oz'):
    loc_dir = os.fsencode(loc_path)
    loc_files = os.listdir(loc_dir)
    for file in loc_files:
        loc_var = os.path.join(loc_dir,file)
        chanlocs = getChanLocs(loc_var)
        if chan in chanlocs:
            chan_ind = chanlocs.index(chan)
        else:
            print(files[0], 'did not have',chan)
            input('understood?')
            chan_ind = -1
        yield chan_ind











