# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 21:48:16 2019

@author: Osamah Hussain
"""

import numpy as np
import matplotlib.pyplot as plt 

from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_feeder import DataFeeder


def plot_decision_regions(X, y, classifier, resolution=.02, test_idx=None):
    """
        Function to print decision boundaries

        :param X - Dataset feature column
        :param y - Dataset feature target
        :param classifier - Model classifier
        :param test_idx - Array IDs of test data
    """
    # setup marker generator & color map
    plt.figure()
    markers = ('x', 'o')
    colors = ('red', 'blue')

    # calculate and plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=.35, cmap=ListedColormap(colors=colors[:len(np.unique(y))]))
    plt.xlim(xx1.min(), xx2.max())
    plt.ylim(xx2.min(), xx2.max())

    # scatter plot all values of the data sets
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolors='black')
    if test_idx:
        # circle test data
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolors='black',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=100,
                    label='test set')

def std_test_train_split(features, target, random_state=1, n_neighbors=5, metric='euclidean', test_size=.3, algorithm='auto'):
    """
        Function to run standard training and testing data split model training and validation

        :param features - Dataset feature columns
        :param target - Dataset target column
        :param random_state - Number for random generator
        :param n_neighbors - Number of neightbors model should use
        :param metric - Classifier distance metric
        :param test_size - Percentage of data to leave for the testing and validation
        :param algorithm - Algorithm to compute the nearest neighbours. If set to 'auto', classifier will automatically determine
                    what algorithm is best suited for the dataset
    """
    # split features and target into training and testing datasets
    # stratify - paramter that makes sure data is split evenly
    features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=random_state, test_size=test_size, stratify=target)
    # initialize k-nearest neighbors classifier
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)
    # train classifier
    clf.fit(features_train, target_train)
    # predict target from provided testing features
    target_predict = clf.predict(features_test)
    # calculate accuracy score of the model in percentages
    acc = accuracy_score(target_test, target_predict) * 100
    print('Accuracy score (train_test_split=%.2f): %.2f' % (test_size *100, acc))
    # plot decission regions if you have 2 dimensional feature dataset
    if len(features.columns) == 2:
        X = np.vstack((features_train, features_test))
        y = np.hstack((target_train, target_test))
        plot_decision_regions(X, y, clf, test_idx=range(len(features_train) - 1, len(features) - 1))
    # calculate confusion matrix
    cm = confusion_matrix(target_test, target_predict)
    plot_confusion_matrix(cm)

    # calculate F-1 score, precision and recall
    calculate_f1_score(target_test, target_predict)

def plot_confusion_matrix(cm_res):
    """
        Function to plot confusion matrix

        :param cm_res - Array containing confusionmatrix resuslts
    """
    fig, ax = plt.subplots(figsize=(5,5))
    ax.matshow(cm_res, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm_res.shape[0]):
        for j in range(cm_res.shape[1]):
            ax.text(x=j, y=i,s=cm_res[i, j], va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')

def cross_validation(features, target, n_neighbors=5, n_folds=5):
    """
        Function to cross validation algorithm over K-nearest neighbor classifier

        :param features - Dataset feature columns
        :param target - Dataset target column
        :param n_neighbors - Number of neightbors model should use
        :param n_folds - Number of folds to perform
    """
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    cv_scores = cross_val_score(clf, features, target, cv=n_folds)
    # print each cv score (accuracy) and average them
    print('Cross Validation Scores Mean: %.2f' % (np.mean(cv_scores) * 100))


def find_best_params(features, target, n_folds=5):
    """
        Function to find best hyperparameters for K-nearest neighbors classifier

        :param features - Dataset feature columns
        :param target - Dataset target column
        :param n_folds - Number of folds to perform during the search
    """
    # prepare hyperparameter set
    params = {'n_neighbors': np.arange(1, 25)}
    # init the search algorithm
    clf = GridSearchCV(KNeighborsClassifier(), params,cv=n_folds)
    # fit the dataset
    clf.fit(features, target)
    # return best hyperparameters
    return clf.best_params_


def calculate_f1_score(y_test, y_pred):
    """
        Function calculate precision, recall and F1-score of your model
        This helps to identify if the model good. You may want to include
        that in you report
    :param y_test: Testing data set
    :param y_pred: Actually predicted data set
    """
    print('# Running precision, recall and F1-score')
    print('# F1-Score:\t\t%.2f' % (f1_score(y_test, y_pred, average="macro") * 100))
    print('# Precision:\t\t%.2f' % (precision_score(y_test, y_pred, average="macro") * 100))
    print('# Recall:\t\t%.2f' % (recall_score(y_test, y_pred, average="macro") * 100))

def plot_hist(data, num_bins=2, facecolor='blue', xlabel='', ylabel='', edgecolor='black', title='', xlim=None):
    plt.figure()
    plt.hist(data, bins=num_bins, facecolor=facecolor,
             alpha=.2, edgecolor=edgecolor)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if xlim is not None:
        axes = plt.axes()
        axes.set_xlim(xlim)

def main():
    # init data feeder
    df = DataFeeder()
    # get pre-processed features and target
    features, target = df.get_data()

    plot_hist(target, xlabel='Diagnosis', ylabel='Patient Records', title='Patient Diagnosis Distribution', xlim=['M', 'B'])

    # run PCA to reduce data dimensionality
    # run several times to campare classifier prediction accuracy and error rate
    # features = df.pca(n_components=2)
    # features = df.pca(n_components=4)
    features = df.pca(n_components=10)

    # find best hyperparameter
    n_neighbors = find_best_params(features, target)['n_neighbors']
    print("Best number of neighbors: %d" % n_neighbors)
    # run train_test_split
    std_test_train_split(features, target, n_neighbors=n_neighbors)
    # run cross validation
    cross_validation(features, target, n_neighbors=n_neighbors)
    # show all graphs
    plt.show()


if __name__ == '__main__':
    main()