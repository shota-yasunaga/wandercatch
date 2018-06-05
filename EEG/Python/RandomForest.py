import util
import numpy as np
import matplotlib.pyplot as plt
from mat2python import data

from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    X,y = data('/Users/macbookpro/Dropbox/College/6th Semester/Network Science/Project/Network_Science_Project/Secret')
    # k_fold = util.KFolder(X,y,1) # Don't get confused. k_fold is ieterator
    # print(k_fold)
    X = scale(X)
    k_fold = util.splitData(X,y,n_split=5)

    
    rf = RandomForestClassifier(class_weight='balanced',bootstrap = False, min_samples_leaf = 1, n_estimators =  5000, max_features = 'sqrt', min_samples_split = 5, max_depth = 100, criterion="entropy")
    training_list = []
    test_list = []
    for X_train, y_train, X_test, y_test in k_fold:
        rf.fit(X_train,y_train)
        pred = rf.predict(X_test)
        rf.fit(X_train,y_train)
        y_pred = rf.predict(X_test)
        training_accuracy=accuracy_score(y_train,rf.predict(X_train))
        test_accuracy = accuracy_score(y_test,y_pred)
        print('------------------')
        print(rf)
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


if __name__ == "__main__":
    main()

