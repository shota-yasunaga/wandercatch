from SVM import *

import numpy as np
from mat2python import getWholeFeatures,getSubsampledFeatures,getFeaturesItr
from freq_power_comparison import getNormalizedFeatures

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score,accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def ind2chan_freq(ind,nChans,nFreq,nTimeBins,min_freq,max_freq):
    '''
    Get the value for the channel and frequency given the index
    The indexing system should be (chans, freq,time_bins)
    inuput: input...        
    '''
    if nTimeBins == 0:
        chan = ind // nFreq + 1 # indexing of channels start from 1
        freq_ind = ind - nFreq*(chan -1)
        freq = np.linspace(min_freq,max_freq,nFreq)[freq_ind]
        return chan, freq
    else:
        time_bin  = ind // (nChans * nFreq)
        chan_freq = ind - time_bin*(nChans*nFreq)
        chan      = chan_freq // nFreq + 1
        freq_ind  = chan_freq  - (chan - 1)*nFreq
        freq = np.linspace(min_freq,max_freq,nFreq)[freq_ind]
        return chan, freq, time_bin

def flatten_features(X):
    '''Helper function'''        
    shape = X.shape
    X = X.reshape(shape[0],-1)
    return X


ppt_list = [301,302,303,305,308,311,312,313,314,317,322,323 ]
C_list   = [0.0001 for i in ppt_list]
num_sub_samples = 10


titles = ['S'+str(ppt) for ppt in ppt_list]
suptitle = 'coefficients'
score_list = []

for (ppt,C) in zip(ppt_list,C_list):
    print('==================')
    print('Participant: ',ppt)

    ppt = str(ppt)
    # Input
    on_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Features/On/freq_ONpPR_ffefspm_S'+ppt+'.mat'
    mw_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Features/MW/freq_WMpPR_ffefspm_S'+ppt+'.mat'

    # Output
    saving_var_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Accuracy/accuracy'
    saving_plot_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Plot_Classification/s'+ppt+'/'+str(num_sub_samples)+'Subsamples/accuracy'

    clf = SVC(kernel='linear',C=C)
    sample = getSubsampledFeatures(on_path,mw_path,max_freq = 40,num_fold = 1)
    X, y = next(sample)
    print(np.array(X).shape)
    clf.fit(flatten_features(X),y)

    coefs_inds = np.argsort(clf.coef_)[0]
    
    nElectrodes = 64
    nFrequency  = 401
    nTimeBins   = 0

    chan_list   = []
    freq_list   = []
    print('Cotributing to On task')
    for i in range(20):
        ind = coefs_inds[i]
        chan,freq = ind2chan_freq(ind,nElectrodes,nFrequency,nTimeBins, 0,40)
        chan_list.append(chan)
        freq_list.append(freq)
        print('  ', (chan,freq))
    
    print('Contributing to the MW')
    for i in range(1,20):
        ind = coefs_inds[-i]
        chan,freq = ind2chan_freq(ind,nElectrodes,nFrequency,nTimeBins, 0,40)
        chan_list.append(chan)
        freq_list.append(freq)
        print('  ', (chan,freq))

chan_list.sort()
freq_list.sort()
print(chan_list)
print(freq_list)
