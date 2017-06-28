# coding=gbk
'''
Created on 2016年6月4日

@author: bryan
'''

import time
from sklearn import metrics
import pickle as pickle
import pandas as pd
import numpy as np


# Multinomial Naive Bayes Classifier  
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier  
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier  
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l1')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier  
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier  
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=1000, max_depth=7)
    model.fit(train_x, train_y)
    return model


# SVM Classifier  
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='linear', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='linear9+', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


def read_data(data_file):
    data = pd.read_csv(data_file)
    train = data[:int(len(data) * 0.9)]
    test = data[int(len(data) * 0.9):]
    train_y = train.label
    train_x = train.drop('label', axis=1)
    test_y = test.label
    test_x = test.drop('label', axis=1)
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':

    thresh = 0.5
    model_save_file = None
    model_save = {}

    print('reading training and testing data...')
    train1 = np.loadtxt('Pre_Process_Train.csv', delimiter=',')
    train_x = train1[:, 1:]
    train = np.loadtxt('dota2Train.csv', delimiter=',')
    train_y = train[:, 0]
    test1 = np.loadtxt('Pre_Process_Test.csv', delimiter=',')
    test_x = test1[:, 1:]
    test = np.loadtxt('dota2Test.csv', delimiter=',')
    test_y = test[:, 0]

    # train = np.loadtxt('dota2Train.csv', delimiter=',')
    # train_x = train[:, 1:117]
    # train_y = train[:, 0]
    #
    # for x in train_x:
    #     for i in range(3,116):
    #         x[i] = (float(x[i]) + 1) / 2.0

    # test = np.loadtxt('dota2Test.csv', delimiter=',')
    # test_x = test[:, 1:117]
    # test_y = test[:, 0]
    #
    # for x in test_x:
    #     for i in range(3,116):
    #         x[i] = (float(x[i]) + 1) / 2.0

    test_classifiers = ['KNN', 'LR', 'RF', 'DT', 'GBDT', 'SVM', 'SVMCV']
    # test_classifiers = ['LR']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   }

    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        precision = metrics.precision_score(test_y, predict)
        recall = metrics.recall_score(test_y, predict)
        print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(test_y, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))

    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))