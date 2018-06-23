import sys
sys.path.append("..") # Adds higher directory to python modules path.
from util import cd
import util
import numpy as np
from mat2python import getWholeFeatures,getSubsampledFeatures,getFakeData
from plot_methods import plot_scatters
from scipy.io import savemat

# Plots
import matplotlib.pyplot as plt
# Sklearn 
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

class ScoreSummary(object):
    """Class to store the training scores and testing scores"""
    def __init__(self, dimensions =[]):#training_list=[],test_list=[]
        super(ScoreSummary,self).__init__()
        self.training_list = []
        self.test_list     = []
        self.dimensions    = dimensions 
        self.shape         = None # Not working now
        self.save_plot_bool= False
        self.save_plot_path= None
    def append(self,score_like):
        '''append the training_list and test_list
            input: score_like: it can be Score Summary or list(can be multilayer)
            output: None. It changes the self.training_list and self.test_list
        '''
        if isinstance(score_like,ScoreSummary):
            append_training,append_test = score_like.getTuple()

            self.training_list.append(append_training)
            self.test_list.append(append_test)
            self.dimensions+=(score_like.dimensions)
            self.shape = np.array(self.test_list).shape
        else:
            append_training,append_test = score_like
            self.training_list.append(append_training)
            self.test_list.append(append_test)
            self.shape = np.array(self.test_list).shape

    def getTuple(self):
        '''
        '''
        return np.array(self.training_list),np.array(self.test_list)
    def std(self,axis = None):
        self.training_std = np.std(self.training_list,axis = axis)
        self.test_std     = np.std(self.test_list,axis = axis)
        return self.training_std, self.test_std
    def mean(self,axis = None):
        self.training_mean = np.mean
        self.test_mean     = np.mean(self.training_list)
    def report(self,means,stds):
        '''under development'''
        return self
    def plot(self,x,subplot_dims=[2,3],across_dim=None,clf_dim=None,suptitle=None,titles = '',x_label='Across Dim',y_label='Accuracy',plt_func=plt.errorbar):
        '''It plots the accuracy with standard deviation'''
        # Mean
        if suptitle==None:
            suptitle=str(self.shape)
        axis = tuple(dim for dim in range(len(self.shape)) if (dim != across_dim and dim != clf_dim))
        training_mean = np.mean(self.training_list,axis = axis)
        test_mean     = np.mean(self.test_list,axis = axis)

        # std
        training_std = np.std(self.training_list,axis = axis)
        test_std     = np.std(self.test_list,axis = axis)

        # Plot the figures
        num_figures = subplot_dims[0]*subplot_dims[1]
        for i in range(self.shape[clf_dim]):
            if (i % num_figures) == 0: # Create new figure if it exceeds subplot dimensions
                self.save_plot(suptitle+'-'+titles[i-num_figures]+'-'+titles[i-1])
                plt.figure(figsize=(15,10))
                plt.suptitle(suptitle)
            plt.subplot(*subplot_dims,i%(num_figures)+1)
            plt_func(x,training_mean[:,i],training_std[:,i],label='Training(with std)')
            plt_func(x,test_mean[:,i],test_std[:,i],label='Test(with std)')
            plt.title(titles[i])
            plt.legend()
            plt.x_label=x_label
            plt.x_label=y_label
            plt.ylim((0,1.05))
        self.save_vars(test_mean,suptitle,titles)
    def save_plot_init(self,save_plot_path):
        '''make the self.plot save the plot to save_plot_path'''
        self.save_plot_bool = True
        self.save_plot_path = save_plot_path
    def save_plot(self, title):
        '''helper function called in self.plot(). It saves the plot'''
        if self.save_plot_bool:
            with cd(self.save_plot_path):
                plt.savefig(title+'.png')
    def save_vars_init(self,save_vars_path):
        '''make the self.plot save the variables(test_mean) to save_vars_path'''
        self.save_vars_bool = True
        self.save_vars_path = save_vars_path
    def save_vars(self, test_mean,suptitle,titles):
        '''Helper method called in self.plot(). It saves the variables'''
        if self.save_vars_bool:
            with cd(self.save_vars_path):
                saving_dic = {titles[i]: test_mean[:,i] for i in range(len(titles))}
                savemat(suptitle+'.mat',saving_dic)




class ClfItrFolder(object):
    """Common elements for all of the clf folders
       it also has some interface (self.score)
    """
    def __init__(self):
        super(ClfItrFolder, self).__init__()
        self.score_list = ScoreSummary([type(self)])
        self.verbose = False
    def init_score_list(self):
        self.score_list=ScoreSummary([type(self)])
    def score(self,X_train,y_train,X_test,y_test):
        return self.score_list



class acrossDimClassifier(ClfItrFolder):
    """docstring for acrossDimClassifier"""
    def __init__(self, clf, as_func_dim):
        super(ClfItrFolder, self).__init__()
        self.clf = clf
        if as_func_dim == 1:
            self.get_local = lambda X,i: X[:,i]
            self.itr       = lambda X: range(len(X[0]))
        elif as_func_dim == 2:
            self.get_local = lambda X,i: X[:,:,i]
            self.itr       = lambda X: range(len(X[0][0]))
        elif as_func_dim == 3:
            self.get_local = lambda X,i: X[:,:,:,i]
            self.itr       = lambda X:range(len(X[0][0][0]))
        elif as_func_dim == 0: # Not iterate
            self.get_local = lambda X,i: X
            self.itr       = [0]
        else:
            error('Expand the function to supper bigger dimension (performance_subsamples_as_func)')
    def flatten_features(self,X):
        '''Helper function'''        
        shape = X.shape
        X = X.reshape(shape[0],-1)
        return X
    def scale_flatten_features(self,X_train,X_test,i):
        '''Helper function'''
        training_length = X_train.shape[0]
        test_length     = X_test.shape[0]
        X = np.vstack((X_train,X_test))
        X_local = self.get_local(X,i)
        X_local = self.flatten_features(X_local)
        X_local = scale(X_local)
        
        X_train = X_local[0:training_length,:]
        X_test  = X_local[training_length:,:]
        return X_train,X_test

    def score(self,X_train,y_train,X_test,y_test):
        self.init_score_list()
        for i in self.itr(X_test):
            X_train_itr,X_test_itr = self.scale_flatten_features(X_train,X_test,i)
            # print(self.clf.score(X_train_itr,y_train,X_test_itr,y_test))
            self.score_list.append(self.clf.score(X_train_itr,y_train,X_test_itr,y_test))
        return self.score_list


class clfFolder(ClfItrFolder):
    """docstring for clfFolder"""
    def __init__(self, clfs):
        super(ClfItrFolder, self).__init__()
        self.clfs= clfs
        self.accuracy_measure_func = accuracy_score
        self.verbose = False
    def fit(self,X,y):
        clf_training = []
        clf_test     = []
        for clf in self.clfs:
            clf.fit(X,y)
    def predict(self,X,y):
        predictions = []
        for clf in self.clfs:
            predictions.append(clf.predict(X))
        return predictions
    def score(self,X_train,y_train,X_test,y_test):
        self.init_score_list()
        clfs_scores = [] #(clfs,:)
        for clf in self.clfs:
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            # training_accuracy=f1_score(y_train,clf.predict(X_train),average="weighted")
            # test_accuracy = f1_score(y_test,y_pred,average="weighted")
            training_accuracy = self.accuracy_measure_func(y_train,clf.predict(X_train))
            test_accuracy     = self.accuracy_measure_func(y_test,y_pred)
            self.score_list.append((training_accuracy,test_accuracy))
            if self.verbose:
                print('------------------')
                print(clf)
                print('training accuracy score')
                print(training_accuracy)
                print('test accuracy score')
                print(test_accuracy)
                print('Confusion Matrix')
                print(confusion_matrix(y_test,y_pred))
        return self.score_list


class subSampler(ClfItrFolder):
    """docstring for subSampler"""
    def __init__(self, clf,num_sub_samples):
        super(ClfItrFolder, self).__init__()
        self.clf = clf
        self.num_sub_samples = num_sub_samples
        self.subsample_func  = getSubsampledFeatures
        self.sub_sample      = None

    def init_sub_sample(self,on_path,mw_path,max_freq):
        self.sub_sample = self.subsample_func(on_path,mw_path,max_freq,self.num_sub_samples)
    def score(self):
        self.init_score_list()
        if self.sub_sample == None:
            error('You need to run self sub_sample first to run score')
        
        if self.progress:
            print('|'+' '*self.num_sub_samples+'|')
            itr = 0
        for X,y in self.sub_sample:
            self.score_list.append(self.clf.score(X,y))
            if self.progress:
                itr += 1
                print('|'+ itr*'*' + ' '*(self.num_sub_samples-itr) + '|')
        return self.score_list
    def progress_bar(self):
        self.progress = True
        
class kFolder(ClfItrFolder):
    """docstring for kFolder"""
    def __init__(self, clf,num_split):
        super(ClfItrFolder, self).__init__()
        self.clf = clf
        self.num_split = num_split
        self.k_fold_func = util.splitData

    def score(self,X,y):
        self.init_score_list()
        k_fold = self.k_fold_func(X,y,self.num_split)
        for X_train,y_train,X_test,y_test in k_fold:
            score = self.clf.score(X_train,y_train,X_test,y_test)
            self.score_list.append(score)
        return self.score_list

def errorfill(x,y,error,label=None,label_fill=None):
    print(np.array(x).shape)
    print(y.shape)
    print(error.shape)
    plt.fill_between(x,y-error,y+error,label=label)
    plt.plot(x,y,'k-')

def main():
    # Input
    on_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Features/On/freq_ONpPR_ffefspm_S317.mat'
    mw_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Features/MW/freq_WMpPR_ffefspm_S317.mat'
    # Output
    saving_var_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Accuracy'
    saving_plot_path = '/Volumes/SHard/Tsuchiya_Lab_Data/Probes/Unprocessed/Plot_classification'

    C = [0.005,5,5]
    C = [str(C[0]),str(C[1]),str(C[2])]
    num_fold = 10

    # TODO:
        # Grid Search
        # all three dimensions
        # with everything
    titles = []
    clfs = []
    for kernel in ['linear','poly','rbf']:
        for C in [0.001,0.005,0.1,0.5,1,5]:
            titles.append(kernel+ '(C='+str(C)+')')
            clfs.append(SVC(kernel=kernel,C=C))
    clf_fold = clfFolder(clfs)
    


    # Train and test the classifiers
    scores_list = []
    for dim in [1,2,3]:
        adc = acrossDimClassifier(clf_fold,dim)
        k_fold   = kFolder(adc,num_split=5)
        subs = subSampler(k_fold,num_sub_samples=10)
        subs.init_sub_sample(on_path,mw_path,max_freq = 40)
        subs.progress_bar()
        scores = subs.score()
        scores.save_plot_init(saving_plot_path)
        scores.save_vars_init(saving_var_path)
        scores_list.append(scores)


    # plot all of the scores
    same_sup  = 'Classification across '
    suptitles = [same_sup+'channels',same_sup+'frequency',same_sup+'time bins']
    x_labels = ['Channels','Frequency (Hz)','Time bins before probe(ms)']
    y_label  ='Accuracy'
    xs  = [list(range(1,65)),np.linspace(0,40,81),np.linspace(1000,0,5)]
    plt_funcs = [errorfill,errorfill,plt.errorbar]

    for i in range(len(scores_list)):
        suptitle = suptitles[i]
        x  = xs[i]
        x_label  = x_labels[i]
        scores_list[i].plot(x,across_dim=2,clf_dim=3,suptitle=suptitle,titles = titles,x_label=x_label,y_label=y_label,plt_func=plt_funcs[i])


    # clf_fold = clfFolder([lin_clf,pol_clf,rbf_clf])
    # adc = acrossDimClassifier(clf_fold,1)
    # k_fold = kFolder(adc,num_split=5)
    # subs = subSampler(k_fold,num_sub_samples=10)
    # subs.init_sub_sample(on_path,mw_path,max_freq=30)
    # scores = subs.score()

    # print(scores.shape)

    # suptitle='Classification across channels'
    # titles = ['Linear (C=0.005)','Polynomial (C=5)','RBF (C=5)']
    # x_label = 'Channels'
    # y_label = 'Accuracy'
    # scores.plot(list(range(64)),across_dim=2,clf_dim=3,suptitle=suptitle,titles = titles,x_label=x_label,y_label=y_label,plt_func=errorfill)




if __name__ == "__main__":
    main()





