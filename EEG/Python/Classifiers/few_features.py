import sys
sys.path.append("..") # Adds higher directory to python modules path.
from SVM import *
import numpy as np
from decimal import Decimal

from mat2python import getPartialSubsampledFeatures,getSubsampledFeatures,flatten_features

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score,accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron

# First attempt -> look at the coefficients -> chose the features that have high/low coefficients

ppt_list = [301,302,303,305,308,311,312,313,314,317,322,323]

odd_list   = [ppt for ppt in ppt_list if ppt % 2 == 1]
even_list  = [ppt for ppt in ppt_list if ppt % 2 == 0]


# ppt_list = [322,323 ]
num_sub_samples = 10
C_for_all_features = [10**(-7),10**(-6),0.00001,0.0001,0.001,0.005,0.1,0.5,1,5,10,50]


def getFreqs(nFreqs, minFreq, maxFreq, freqs):
    freq_vec = list(np.linspace(minFreq,maxFreq,nFreqs))
    takeClosest = lambda num,collection:min(collection,key=lambda x:abs(x-num))

    closest_freqs = [takeClosest(f,freq_vec)for f in freqs]

    freq_inds = [freq_vec.index(f) for f in closest_freqs]

    return lambda features: features[:,:,freq_inds]

big_score = None



for ppt in odd_list:
    ppt = str(ppt)
    # Input                                                                                 raw_ONbpTR_ffefspm_S301.mat
    on_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Features/On/freq_ONpPR_ffefspm_S'+ppt+'.mat'
    mw_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Features/MW/freq_WMpPR_ffefspm_S'+ppt+'.mat'

    # Output
    # saving_var_path  = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Accuracy/accuracy'
    saving_plot_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Plot_Classification/SelectedFeatures'
    titles = []

    minFreq  = 0
    maxFreq = 40
    nFreqs   = 401


    foi = list(np.linspace(8, 13, 26))+ [7.4, 7.5,7.6, 15.1,15.0,15.2, 23.9,24,24.1] # Frequency of interest (odd)
    # foi = list(np.linspace(8, 13, 26)) + [5.9,6,6.1, 14.9, 15,15.1,29.9,30,30.1] # Frequeicy of interest(even)

    partial_func = getFreqs(nFreqs,minFreq,maxFreq,foi)

    ########################
    # Get the coefficients #
    ########################
    clf = SVC(kernel='linear',C=0.005)
    sample = getSubsampledFeatures(on_path,mw_path,max_freq = 40,num_fold = 1)
    X, y = next(sample)
    clf.fit(flatten_features(X),y)

    # the index of coefficient based on the ahbsolute magnitude of coefficients
    coefs_inds = np.argsort(abs(clf.coef_))[0]
    
    #########################################
    # Gridsearch with respect to dimensions #
    #########################################

    clfs = []
    for kernel in ['linear']:
        for C in [0.00001,0.0001,0.001,0.1]:
            titles.append(kernel+ '(C='+str(C)+')')
            clfs.append(SVC(kernel=kernel,C=C))
    clf_fold = clfFolder(clfs)
    clf_fold.setAccuracyFun(accuracy_score,normalize=True)

    # Train and test the classifiers
    scores_list = []
    # for dim in [1,2]:
    max_inds = [25664]
    # for max_ind in max_inds:
    dim  = 0
    adc = acrossDimClassifier(clf_fold,dim)
    k_fold   = kFolder(adc,num_split=5)
    subs = subSampler(k_fold,num_sub_samples=10)
    subs.set_subsample_func(getPartialSubsampledFeatures)
    subs.init_sub_sample(on_path,mw_path,max_freq = 40,feature_type='freq', partial_func = partial_func)
    # subs.progress_bar()
    scores = subs.score() 
    scores.save_plot_init(saving_plot_path)
    # if(dim == 1):
    #     scores.save_vars_init(saving_var_path)
    scores_list.append(scores)
    

    # # plot all of the scores
    same_sup  = 'S'+ppt+'  '
    # suptitles = [same_sup+ str(max_ind) + ' Features' for max_ind in max_inds]
    suptitles = [same_sup  + ' Selected Features']
    # x_labels = ['Channels','Frequency (Hz)','Time bins before probe(ms)']
    x_labels = ['Classifier'] * len(max_inds)
    y_label  ='Accuracy'
    xs  = [0] * len(max_inds)
    plt_funcs = [plt.errorbar] * len(max_inds)

    # for i in range(len(scores_list)):
    #     suptitle = suptitles[i]
    #     x  = xs[i]
    #     x_label  = x_labels[i]
    #     scores_list[i].plot(x,subplot_dims=[1,4],across_dim=2,clf_dim=3,suptitle=suptitle,titles = titles,x_label=x_label,y_label=y_label,plt_func=plt_funcs[i])

    if big_score == None:
        big_score = scores
        train_list,testing_list = big_score.getTuple()
        big_score.training_list,big_score.test_list = [train_list], [testing_list]
    else:
        big_score.append(scores)
        print(scores.shape)
        print(big_score.shape)
        print('+++++++++++++++++')
big_score.plot([str(ppt) for ppt in odd_list], subplot_dims=[1,4], across_dim=0,clf_dim=4,suptitle='PptsTogether',titles = titles, x_label = 'Ppt Num', y_label='Accuracy',plt_func=plt.errorbar)