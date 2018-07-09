import os

import numpy as np
import scipy.io as io # data 
from collections import Counter
# scikitlearn
import sklearn.utils as sku
from sklearn.utils import resample
import sklearn.preprocessing as prep
from sklearn.decomposition import PCA




##################################
# Related to Feature Extractions #
##################################
def getRawValues(file_path):
    f = open(file_path,'rb')
    data = np.array(io.loadmat(f)['spectra'])
    f.close()
    return data

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


def getSubsampledFeatures(cond0_path,cond1_path,max_freq=-1,num_fold=1,feature_type = 'freq'):
    if feature_type == 'freq':
        features0,freqVec = readOneFeatures(cond0_path,max_freq)
        features1,freqVec = readOneFeatures(cond1_path,max_freq)
    
    elif feature_type == 'raw':
        features0 = getRawValues(cond0_path)
        features1 = getRawValues(cond1_path)
    
    num_samples = min(len(features0),len(features1))
    labels  = np.concatenate((np.zeros(num_samples),np.ones(num_samples)))
    for i in range(num_fold):
        resampled0 = resample(features0,replace=False,n_samples=num_samples)
        resampled1 = resample(features1,replace=False,n_samples=num_samples)
        features   = np.vstack((resampled0,resampled1))

        yield features,labels

def flatten_features(X):
    '''Helper function for getPCA'''        
    shape = X.shape
    X = X.reshape(shape[0],-1)
    return X

def getNormalizedFeatures(cond0_path,cond1_path,max_freq,num_fold):
    '''generator that takes iterable.
    input featureItr: should give features, labels in each iteration
    '''
    from freq_power_comparison import normalize_power

    featureItr = getSubsampledFeatures(cond0_path,cond1_path,max_freq,num_fold)
    for features,labels in featureItr:
        yield normalize_power(features),labels

def getPCASubsampledFeatures(cond0_path,cond1_path,max_freq,num_fold, n_components= 100):
    '''generator for PCAed features and labels'''
    pca = PCA(n_components = n_components)
    for features, labels in getSubsampledFeatures(cond0_path,cond1_path,max_freq,num_fold):
        features = flatten_features(features)
        features = pca.fit_transform(features)
        yield features, labels

def getFeaturesItr(cond0_path,cond1_path,max_freq,num_fold):
    features0,freqVec = readOneFeatures(cond0_path,max_freq)
    features1,freqVec = readOneFeatures(cond1_path,max_freq)

    f0_num = features0.shape[0]
    f1_num = features1.shape[0]

    labels  = np.concatenate((np.zeros(f0_num),np.ones(f1_num)))
    for i in range(num_fold):
        features   = np.vstack((features0,features1))

        yield features,labels

def getWholeFeatures(cond0_path,cond1_path,max_freq, seperate = False):
    '''
    get the features based on paths
    Input:
        cond0_path: path to condition0
        cond1_path: path to condition1
        max_freq:   maximum frequency to consider
        seperate:   If the return should be condition-seperated or not
    Output: 
        if seperate:
            features0,features1 ... feautures for cond0,1
            labels0,labels1     ... labels or cond0,1
        else:
            features
            labels
    '''
    features0,freqVec = readOneFeatures(cond0_path,max_freq)
    features1,freqVec = readOneFeatures(cond1_path,max_freq)
    
    features = np.vstack((features0,features1))
    labels = np.concatenate((np.zeros(len(features0)),np.ones(len(features1))))
    if len(labels) != len(features):
        print('labels: '+str(len(labels)))
        print('features: '+str(len(features)))
    if seperate:
        return features0,features1,np.zeros(len(features0)),np.ones(len(features1))
    else:
        return features,labels

def getAllPptFeatures(cond0_folder,cond1_folder,max_freq,ppt_list=None, normalize = False):
    '''
    summary: 
    Input: 
        cond0_folder: path to the folder that contains data for condition 0
        cond1_folder: path to the folder that contains data for condition 1
        max_freq: maximum frequency to consider
        ppt_list: participants to consider. None means all of the participants under the folder (default None)
        normalize: return normalized power(power/sum of the power of all of the band) (default False)
    output: 

    Warning: this method is dependent on that fact that the file name ends with ppt_num.mat
             where ppt_num is the participant number (ex, filename313.mat)
    '''
    c0_itr = folderIterator(cond0_folder)
    c1_itr = folderIterator(cond1_folder)
    f0_list = []
    f1_list = []
    count0 = 0
    count1 = 0
    # TODO: make it simple?
    for c0,c1 in zip(c0_itr,c1_itr):
        ppt_num = int(c0[-7:-4])
        if (ppt_num in ppt_list) or (ppt_list ==  None):
            f0,f1,l0,l1 = getWholeFeatures(c0,c1,max_freq,seperate=True)
            f0_list.append(f0) # TODO: make sure this is right.
            f1_list.append(f1)        
            count0 += len(l0)
            count1 += len(l1)
    f0_list = np.vstack(f0_list)
    f1_list = np.vstack(f1_list) 
    if normalizer:
        from freq_power_comparison import normalize_power
        return normalize_power(f0_list),normalize_power(f1_list),count0,count1
    else:
        return f0_list,f1_list,count0,count1



def getSubsampledAllPpt(cond0_folder,cond1_folder,max_freq,num_fold,ppt_list=None, normalize = False):
    features0,features1,count0,count1 = getAllPptFeatures(cond0_folder,cond1_folder,max_freq,ppt_list= ppt_list,normalize = normalize)
    # count each label
    for features, labels in subSampledFeaturesHelper(features0,feature1,count0,count1,num_fold):
        yield features, labels

def subSampledFeaturesHelper(features0,feature1,count0,count1, num_fold):
    num_samples = min(count0,count1)

    labels  = np.concatenate((np.zeros(num_samples),np.ones(num_samples)))
    for i in range(num_fold):
        resampled0 = resample(features0,replace=False,n_samples=num_samples)
        resampled1 = resample(features1,replace=False,n_samples=num_samples)
        features   = np.vstack((resampled0,resampled1))
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



##########
# Others #
##########
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



