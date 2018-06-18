import sys
sys.path.append("..") # Adds higher directory to python modules path.
from util import cd
import util
import numpy as np
import matplotlib.pyplot as plt
from mat2python import getWholeFeatures,getSubsampledFeatures,getFakeData
from plot_methods import plot_scatters
from scipy.io import savemat

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


def tune(X_train, y_train, scoring):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

    fracNeg = len(y_train[y_train == 0])/float(len(y_train))
    weight = (1-fracNeg)/float(fracNeg) 
    class_weight = {1:1, 0:weight}

    rf = RandomForestClassifier(class_weight=class_weight, criterion="entropy")
    # automatically does stratified kfold
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1, scoring=scoring)
    rf_random.fit(X_train, y_train)
    return rf_random.best_params_, rf_random.best_score_

class chooseFeatureSVM(object):
    """docstring for chooseFeatureSVM"""
    def __init__(self, clfs, as_func_dim):
        super(chooseFeatureSVM, self).__init__()
        self.arg = arg
        self.clfs= clfs
        self.as_func_dim = as_func_dim
        self.accuracy_mean  = []
        self.accuracy_whole = 
    def fit(self,X,y):
        run_multiple_clfs()

    def run_multiple_clfs(X_train,k_fold,accuracy_measure_func=accuracy_score,verbose=True,return_type='mean',**kwargs):
    clf_training = []
    clf_test     = []
    for clf in self.clfs:
        clf.fit()
        training,test = k_fold_loop(clf,k_fold,accuracy_measure_func,verbose,return_type,kwargs)
        clf_training.append(training)
        clf_test.append(test)

    return clf_training,clf_test #(clfs,) if mean, (clfs,num_fold) if all




def k_fold_loop(clf,k_fold,accuracy_measure_func=accuracy_score,verbose=True,return_type = 'mean',**kwargs):
    training_list = []
    test_list     = []
    for X_train,y_train,X_test,y_test in k_fold:
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        training_accuracy = accuracy_measure_func(y_train,clf.predict(X_train),kwargs)
        test_accuracy     = accuracy_measure_func(y_test,y_pred,kwargs)

        training_list.append(training_accuracy)
        test_list.append(test_accuracy)
        if verbose:
            print('------------------')
            print(clf)
            print('training accuracy score')
            print(training_accuracy)
            print('test accuracy score')
            print(test_accuracy)
            print('Confusion Matrix')
            print(confusion_matrix(y_test,y_pred))


    if return_type == 'mean':
        return np.mean(np.array(training_list)),np.mean((np.array(test_list))) #Single values
    elif return_type == 'all':
        return np.array(training_list),np.array(test_list) #(num_fold,)
    else:
        error('return_type',return_type,'not definied')


def run_subSamples():
    pass

def svc_loop(k_fold,verbose=True):
    whole_training = []
    whole_test     = []
    for X_train, y_train, X_test, y_test in k_fold:
        # lin_clf = SVC(kernel='linear',C=0.1,class_weight='balanced')
        # pol_clf = SVC(kernel ='poly', C=1,class_weight='balanced',degree=10)
        # rbf_clf = SVC(kernel='rbf', C=0.1,class_weight='balanced')

        lin_clf = SVC(kernel='linear',C=0.005)
        pol_clf = SVC(kernel ='poly', C=5,degree=3)
        rbf_clf = SVC(kernel='rbf', C=5)
        #clfs = [lin_clf,pol_clf,rbf_clf]
        clfs= [lin_clf,pol_clf,rbf_clf]
        training_list = []
        test_list = []
        for clf in clfs:
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            # training_accuracy=f1_score(y_train,clf.predict(X_train),average="weighted")
            # test_accuracy = f1_score(y_test,y_pred,average="weighted")
            training_accuracy = accuracy_score(y_train,clf.predict(X_train))
            test_accuracy     = accuracy_score(y_test,y_pred)
            training_list.append(training_accuracy)
            test_list.append(test_accuracy)
            if verbose:
                print('------------------')
                print(clf)
                print('training accuracy score')
                print(training_accuracy)
                print('test accuracy score')
                print(test_accuracy)
                print('Confusion Matrix')
                print(confusion_matrix(y_test,y_pred))

        whole_training.append(training_list) #(folds,clfs)
        whole_test.append(test_list)

    whole_training = np.transpose(np.array(whole_training)) #(clfs,folds)
    whole_test = np.transpose(np.array(whole_test))

    if verbose:
        print(whole_test)
        print(whole_training)
    # (clfs,trials)
    return whole_training,whole_test

def run_subSamples(subSamples,as_func_dim,n_fold,verbose=True):
    for X,y in subSamples:
    pass

def performance_subsamples_as_func(subSamples,as_func_dim,n_fold,verbose=False):
    '''
    '''
    sub_training=[]
    sub_test    =[]
    for X,y in subSamples:
        print('subsample shape')
        print(X.shape)
        # First dim: channels 
        chans_training = [] #(chans,clfs)
        chans_test     = []

        if as_func_dim == 1:
            get_local = lambda X: X[:,i]
            itr = range(len(X[0]))
        elif as_func_dim == 2:
            get_local = lambda X: X[:,:,i]
            itr = range(len(X[0][0]))
        elif as_func_dim == 3:
            get_local = lambda X: X[:,:,:,i]
            itr = range(len(X[0][0][0]))
        elif as_func_dim == 0: # Not iterate
            get_local = lambda X: X
            itr = [1]
        else:
            error('Expand the function to supper bigger dimension (performance_subsamples_as_func)')

        for i in itr:
            X_local = get_local(X)

            X_local = flatten_features(X_local)
            X_local = scale(X_local)
            k_fold = util.splitData(X_local,y,n_split=n_fold)

            whole_training,whole_test = svc_loop(k_fold,verbose=verbose)
            chans_training.append(np.mean(whole_training,axis =1))
            chans_test.append(np.mean(whole_test,axis = 1))

        chans_training=np.transpose(np.array(chans_training))  # (clfs,chans)
        chans_test=np.transpose(np.array(chans_test))


        sub_training.append(chans_training) #(subsample,clfs,chans)
        sub_test.append(chans_test)   



    sub_training = np.array(sub_training)
    sub_test     = np.array(sub_test)
    ## Reshaping for plotting purpose
    sub_training=np.mean(sub_training,0) 
    sub_test    =np.mean(sub_test,0)

    print('subtraining...')
    print(sub_training.shape)
    performance = np.vstack((sub_training,sub_test))

    return performance #(trainings+tests,num_iteration)

def flatten_features(X):
    shape = X.shape
    X = X.reshape(shape[0],-1)
    return X

def main():
    # Input
    on_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/New_Features/On/freq_ONpPR_ffefspm_S317.mat'
    mw_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/New_Features/MW/freq_WMpPR_ffefspm_S317.mat'
    # Output
    saving_var_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Accuracy'
    saving_plot_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/New_Plot_classification'

    C = [0.005,5,5]
    C = [str(C[0]),str(C[1]),str(C[2])]
    num_fold = 10

    import pandas as pan
    subSamples = getSubsampledFeatures(on_path,mw_path,40,num_fold=1)
    svc = SVC(kernel='linear')
    c_values = [10**x for x in range(-5,4)]
    parameters = {'C':c_values}
    clf = GridSearchCV(svc, parameters)
    for X,y in subSamples:
        
        X = X[:,25,:,:]
        X = flatten_features(X)
        clf.fit(X,y)
        d_frame = pan.DataFrame(clf.cv_results_)
    print(d_frame)

    #############################
    # First Dimension: Channels #
    #############################

    # subSamples = getSubsampledFeatures(on_path,mw_path,40,num_fold=10)
    subSamples = getSubsampledFeatures(on_path,mw_path,40,num_fold=num_fold)
    chans_performance = performance_subsamples_as_func(subSamples,1,5,verbose=False)

    chans=list(range(1,65))
    titles = ['Linear SVM (C='+C[0] +')','Polinomial SVM (C='+C[1]+')','rbf SVM (C='+C[2]+')']*2
    sub_nums = [231,232,233,234,235,236]
    x_label = 'Channels'
    y_label = 'Classification Performance(accuracy)'
    labels = ['Training']*3+['Test']*3

    plt.figure(figsize=(15,10));
    plt.suptitle('SVM as a function of Channels')

    linear    = list(chans_performance[3])
    polynomial= list(chans_performance[4])
    rbf       = list(chans_performance[5])

    # saving variables to .mat file
    with cd(saving_var_path):
        saving_dic = {'linear':linear,'polynomial':polynomial,'rbf':rbf}
        savemat(C[0]+'-'+C[1]+'-'+C[2]+str(num_fold)+'Folds'+'.mat',saving_dic)
    
    
    # Saving the figure 
    plot_scatters(chans,chans_performance,titles,sub_nums,x_label,y_label,regression =False,labels=labels)
    with cd(saving_plot_path):
        plt.savefig(C[0]+'-'+C[1]+'-'+C[2]+'_'+str(num_fold)+'Folds'+'_chans.png')

    #########################
    # Second dim: Frequency #
    #########################
    subSamples = getSubsampledFeatures(on_path,mw_path,40,num_fold=num_fold)

    freqs_performance = performance_subsamples_as_func(subSamples,2,5,verbose=False)
    

    freqs=np.linspace(0,40,len(freqs_performance[0]))
    
    x_label = 'Frequency'

    plt.figure(figsize=(15,10));
    plt.suptitle('SVM as a function of Frequency')
    print(freqs.shape)
    print(freqs_performance.shape)
    plot_scatters(freqs,freqs_performance,titles,sub_nums,x_label,y_label,regression =False,labels=labels,markersize=3)
    with cd(saving_plot_path):
    #     plt.savefig(C[0]+'-'+C[1]+'-'+C[2]+str(num_fold)+'Folds'+'_freqs.png')
    ###################
    # Third dim: time #
    ###################
    
    # subSamples = getSubsampledFeatures(on_path,mw_path,40,num_fold=10)
    # time_performance = performance_subsamples_as_func(subSamples,3,5,verbose=False)
    

    # time= list(range(5))
    # titles = ['Linear SVM','Polinomial SVM','rbf SVM']*2
    # sub_nums = [231,232,233,234,235,236]
    # x_label = 'Time Window'
    # y_label = 'Classification Performance(accuracy)'
    # labels = ['Training']*3+['Test']*3

    # plt.figure();
    # plt.suptitle('SVM as a function of Time')

    # plot_scatters(time,time_performance,titles,sub_nums,x_label,y_label,regression =False,labels=labels)

    # subSamples = getWholeFeatures(on_path,mw_path,30)
    # whole_performance = performance_subsamples_as_func(subSamples,0,5,verbose=True)
    # plt.show()

if __name__ == "__main__":
    main()











