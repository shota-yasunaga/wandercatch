import os

import numpy as np
import scipy.io as io # data 
from collections import Counter

# scikitlearn
import sklearn.utils as sku
from sklearn.utils import resample
import sklearn.preprocessing as prep




def getFreqValuesVec(file_path):
    '''
    data 
    '''
    f = open(file_path,'rb')
    freqValues = np.array(io.loadmat(f)['spectra'])
    freqVec   = np.array(io.loadmat(f)['freqVec'])
    f.close()
    return freqValues,freqVec

def readOneFeatures(file_path,max_freq = -1):
    '''
    TODO: check if this is right!!!!!
    '''
    f = open(file_path,'rb')
    features = np.array(io.loadmat(f)['features'])
    freqVec   = np.array(io.loadmat(f)['freqVec'])
    freqVec = np.array([round(freq[0],1) for freq in freqVec])
    end_ind = list(freqVec).index(max_freq)
    
    features = features[:,:,0:end_ind+1]
    freqVec  = freqVec[0:end_ind+1]
    f.close()
    return features,freqVec


def getSubsampledFeatures(cond0_path,cond1_path,max_freq,num_fold):
    features0,freqVec = readOneFeatures(cond0_path,max_freq)
    features1,freqVec = readOneFeatures(cond1_path,max_freq)
    
    num_samples = min(len(features0),len(features1))
    labels  = np.concatenate((np.zeros(num_samples),np.ones(num_samples)))
    for i in range(num_fold):
        resampled0 = resample(features0,replace=False,n_samples=num_samples)
        resampled1 = resample(features1,replace=False,n_samples=num_samples)
        features   = np.vstack((resampled0,resampled1))

        yield features,labels

def getFakeData():
    ''' Let's depriciate this. Debugging purpose
    '''
    num_samples=24
    for i in range(5):
        labels  = np.concatenate((np.zeros(num_samples),np.ones(num_samples)))
        features = np.vstack(( np.random.rand(num_samples,64,30,5)*10,np.random.rand(num_samples,64,30,5)*(-10)))
        yield features,labels

def getWholeFeatures(cond0_path,cond1_path,max_freq):
    features0,freqVec = readOneFeatures(cond0_path,max_freq)
    features1,freqVec = readOneFeatures(cond1_path,max_freq)
    
    features = np.vstack((features0,features1))
    labels = np.concatenate((np.zeros(len(features0)),np.ones(len(features1))))
    if len(labels) != len(features):
        print('labels: '+str(len(labels)))
        print('features: '+str(len(features)))

    yield features,labels

def getChanLocs(file_path):
    f = open(file_path,'rb')
    mat_labels = io.loadmat(f)['labels'][0]
    f.close()
    labels = ['']*len(mat_labels)
    for i in range(len(mat_labels)):
        labels[i] = mat_labels[i][0]
    return labels

def getLabels(file_path):
    fid = open(file_path)
    data = np.loadtxt(fid,delimiter=' ',dtype={'names':('epoch','label'),'formats':('i4','S2')},skiprows=1)
    fid.close()
    return list(zip(*data))[1]

def getNumLabels(file_path):
    labels = getLabels(file_path)
    decoder = lambda byte: byte.decode("utf-8")
    labels = list(map(decoder,labels))
    return Counter(labels)

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
    '''
    Genrator constructor. 
    input: path to the folder
    output: generator that generates the full_path to each file under the folder

    ex) 
    for file in folderIterator(folder_path):

    '''
    folder_dir = os.fsencode(folder_path)
    files = os.listdir(folder_dir)
    files.sort()
    for file in files:
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



