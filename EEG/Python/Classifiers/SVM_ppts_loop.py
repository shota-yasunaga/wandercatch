import sys
sys.path.append("..") # Adds higher directory to python modules path.
from SVM import *
import numpy as np
from mat2python import getWholeFeatures,getSubsampledFeatures,getFeaturesItr,getPCASubsampledFeatures

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score,accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron




ppt_list = [301,302,303,305,308,311,312,313,314,317,322,323]
# ppt_list = [322,323 ]
num_sub_samples = 10
C_for_all_features = [10**(-7),10**(-6),0.00001,0.0001,0.001,0.005,0.1,0.5,1,5,10,50]


for ppt in ppt_list:
    ppt = str(ppt)
    # Input
    on_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Features/On/freq_ONpPR_ffefspm_S'+ppt+'.mat'
    mw_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Features/MW/freq_WMpPR_ffefspm_S'+ppt+'.mat'

    # Output
    saving_var_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Accuracy/accuracy'
    saving_plot_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Plot_Classification/s'+ppt+'/'+str(num_sub_samples)+'Subsamples/accuracy'
    titles = []

    #########################################
    # Gridsearch with respect to dimensions #
    #########################################

    clfs = []
    for kernel in ['linear','rbf']:
        for C in [0.001,0.005,0.1,0.5,1,5]:
            titles.append(kernel+ '(C='+str(C)+')')
            clfs.append(SVC(kernel=kernel,C=C))

    for penalty in ['l1', 'l2']:
        for alpha in [0.0001, 0.00001,0.000001]:
            titles.append(penalty + '(Alpha = ' + str(alpha) + ')')
            clfs.append(Perceptron(penalty = penalty,alpha = alpha))
    clf_fold = clfFolder(clfs)
    clf_fold.setAccuracyFun(accuracy_score,normalize=True)


    # Train and test the classifiers
    scores_list = []
    for dim in [1,2]:
        adc = acrossDimClassifier(clf_fold,dim)
        k_fold   = kFolder(adc,num_split=5)
        subs = subSampler(k_fold,num_sub_samples=10)
        subs.set_subsample_func(getPCASubsampledFeatures)
        subs.init_sub_sample(on_path,mw_path,max_freq = 40)
        subs.progress_bar()
        scores = subs.score()
        scores.save_plot_init(saving_plot_path)
        if(dim == 1):
            scores.save_vars_init(saving_var_path)
        scores_list.append(scores)


    # # plot all of the scores
    same_sup  = 'S'+ppt+'  '
    suptitles = [same_sup+'channels',same_sup+'frequency',same_sup+'time bins']
    # x_labels = ['Channels','Frequency (Hz)','Time bins before probe(ms)']
    x_labels = ['Channels','Frequency (Hz)']
    y_label  ='Accuracy'
    xs  = [list(range(1,65)),np.linspace(0,40,401),np.linspace(1000,0,5)]
    plt_funcs = [errorfill,errorfill,plt.errorbar]

    for i in range(len(scores_list)):
        suptitle = suptitles[i]
        x  = xs[i]
        x_label  = x_labels[i]
        scores_list[i].plot(x,across_dim=2,clf_dim=3,suptitle=suptitle,titles = titles,x_label=x_label,y_label=y_label,plt_func=plt_funcs[i])



    #######################################
    # Grid Search for all of the features #
    #######################################
    titles = []
    clfs = []
    for kernel in ['linear','poly','rbf']:
        for C in C_for_all_features:
            titles.append(kernel+ '(C='+str(C)+')')
            clfs.append(SVC(kernel=kernel,C=C))
    clf_fold = clfFolder(clfs,verbose=False)
    clf_fold.setAccuracyFun(accuracy_score,normalize=True)
    
    # Train and test the classifiers
    adc = acrossDimClassifier(clf_fold,0)
    k_fold   = kFolder(adc,num_split=5)
    subs = subSampler(k_fold,num_sub_samples=10)
    subs.set_subsample_func(getPCASubsampledFeatures)
    subs.init_sub_sample(on_path,mw_path,max_freq = 40)
    subs.progress_bar()
    scores = subs.score()
    scores.save_plot_init(saving_plot_path)

    # plot all of the scores
    suptitle = 's'+ppt+' with all of the features'
    y_label  ='Accuracy'
    x_label  ='None'
    x = 0
    scores.plot(x,across_dim=None,clf_dim=3,suptitle=suptitle,titles = titles,x_label=x_label,y_label=y_label,plt_func=plt.errorbar)

    
    #######################
    # Random Forest Model #
    #######################
    saving_plot_path='/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Plot_Classification/s'+ppt+'/'+str(num_sub_samples)+'Subsamples/RandomForest'

    rf = RandomForestClassifier(class_weight='balanced',bootstrap = False, min_samples_leaf = 1, n_estimators =  5000, max_features = 'sqrt', min_samples_split = 5, max_depth = 100, criterion="entropy")

    titles = []
    clfs = []
    for kernel in ['Random Forest']:
        for n_estimators in [100,300,500]:
            titles.append(kernel+ '(N_Estimator='+str(n_estimators)+')')
            rf = f = RandomForestClassifier(class_weight='balanced',bootstrap = True, min_samples_leaf = 1, n_estimators =  n_estimators, max_features = 'sqrt', min_samples_split = 3, max_depth = 5, criterion="entropy")
            clfs.append(rf)
    clf_fold = clfFolder(clfs)
    clf_fold.setAccuracyFun(accuracy_score,average='weighted')
    # Train and test the classifiers
    adc = acrossDimClassifier(clf_fold,0)
    k_fold   = kFolder(adc,num_split=5)
    subs = subSampler(k_fold,num_sub_samples=10)
    subs.set_subsample_func(getPCASubsampledFeatures)
    subs.init_sub_sample(on_path,mw_path,max_freq = 40)
    subs.progress_bar()
    scores = subs.score()
    scores.save_plot_init(saving_plot_path)

    # plot all of the scores
    suptitle = 'Random Forest (S'+ppt 
    y_label  ='Accuracy'
    x_label  ='Classifier'
    x = 0
    scores.plot(x,subplot_dims=[1,3],across_dim=None,clf_dim=3,suptitle=suptitle,titles = titles,x_label=x_label,y_label=y_label,plt_func=plt.errorbar)

    