import util
import numpy as np
import matplotlib.pyplot as plt
from mat2python import data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale



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

def main():
    X,y = data('/Users/macbookpro/Dropbox/College/6th Semester/Network Science/Project/Network_Science_Project/Secret')
    # k_fold = util.KFolder(X,y,1) # Don't get confused. k_fold is ieterator
    # print(k_fold)
    X = scale(X)
    k_fold = util.splitData(X,y,n_split=5)

    training_list = []
    test_list = []
    for X_train, y_train, X_test, y_test in k_fold:
        lin_clf = SVC(kernel='linear',C=0.1,class_weight='balanced')
        pol_clf = SVC(kernel ='poly', C=1,class_weight='balanced',degree=10)
        rbf_clf = SVC(kernel='rbf', C=0.1,class_weight='balanced')

        #clfs = [lin_clf,pol_clf,rbf_clf]
        clfs= [lin_clf]
        for clf in clfs:
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            training_accuracy=accuracy_score(y_train,clf.predict(X_train))
            test_accuracy = accuracy_score(y_test,y_pred)
            print('------------------')
            print(clf)
            print('training accuracy')
            print(training_accuracy)
            training_list.append(training_accuracy)
            print('test accuracy')
            print(test_accuracy)
            test_list.append(test_accuracy)
            print('Confusion Matrix')
            print(confusion_matrix(y_test,y_pred))

    print(training_list)
    print(test_list)
    return 
    #########################
    # With pca              #
    # Can we do better????  #
    #########################
    pca = PCA(n_components = 100,svd_solver='auto')
    pca.fit(np.transpose(X)) # this probably is kinda bad,but rn, I'm doing this.
    print(X.shape)
    print(pca.components_.shape)
    features = np.transpose(pca.components_)
    print('================')
    print(features.shape)

    splitter = util.splitData(features,y,n_split=1)
    
    for X_train, y_train, X_test, y_test in splitter:
        lin_clf = SVC(kernel='linear',C=0.1,class_weight='balanced',)
        pol_clf = SVC(kernel ='poly', C=1,class_weight='balanced',degree=10)
        rbf_clf = SVC(kernel='rbf', C=0.1,class_weight='balanced')

        clfs = [lin_clf,pol_clf,rbf_clf]
        for clf in clfs:
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            training_accuracy=accuracy_score(y_train,clf.predict(X_train))
            test_accuracy = accuracy_score(y_test,y_pred)
            print('------------------')
            print(clf)
            print('training accuracy')
            print(training_accuracy)
            print('test accuracy')
            print(test_accuracy)
            print('Confusion Matrix')
            print(confusion_matrix(y_test,y_pred))
    
    # Will tune later


if __name__ == "__main__":
    main()











