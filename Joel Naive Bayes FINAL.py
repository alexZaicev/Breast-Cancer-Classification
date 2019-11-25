# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:08:47 2019

@author:    Joel Scheven - S15126687
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, LabelBinarizer


import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

# disable unwanted future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class DataFeeder(object):
    """ Data feeding class containing logic for processing data frame values
    for further evaluation
        @:param df - Pandas data frame read from CSV file
        @:param features_ - Data frame feature columns
        @:param target_ - Data frame target column
    """

    def __init__(self):
        # read data file and create data frame
        self.df = None
        self.features_ = None
        self.target_ = None

    def get_data(self, normalize=True):
        """
            Function to get processed features and target data sets
        :param normalize: Normalize data set values with standard deviation
        :return: Separated features and target data sets
        """
        __nOrig = 0
        __nAfter = 0
        if self.features_ is None or self.target_ is None:
            # read CSV file and create data frame
            self.__read_csv()
            __nOrig = len(self.df)
            # remove unknown values
            self.__remove_unknown_values()
            # remove categorical data
            self.__transform_categorical_data()
            # identify and remove outliers
            self.__remove_outliers()
            __nAfter = len(self.df)

            # TODO: visualize data frame before splitting into features

            # split data frame into features and target
            self.features_ = self.df.iloc[:, 2:]
            self.target_ = self.df.iloc[:, 1]
            # normalize feature values
            if normalize:
                self.__normalize()

            __diff = __nOrig - __nAfter
            if __diff > 0:
                print('Data frame truncated to %d instances' % __diff)
        return self.features_, self.target_

    def pca(self, n_components=2, kernel='linear', gamma=None):
        """
            Function to perform Principle Component Analysis for dimensionality reduction
        :param n_components: Number of features (components)
        :param kernel: PCA kernel
        :param gamma: Gamma value for Gaussian Kernel (rbf)
        """
        columns = list()
        for i in range(0, n_components, 1):
            columns.append('pc_%d' % i)
        _pca = KernelPCA(kernel=kernel, gamma=gamma, n_components=n_components)
        # _pca = PCA(n_components=n_components)
        _temp = _pca.fit_transform(self.features_)
        self.features_ = pd.DataFrame(data=_temp, columns=columns)
        return self.features_

    def __read_csv(self):
        """
            Function to read CSV file
        """
        self.df = pd.read_csv('BREAST_CANCER_WISCONSIN.csv', header=0)

    def __remove_outliers(self):
        """
            Function to identify rows in multi-dimensional data frame and remove
            rows that contains them
             * For each column, first it computes the Z-score of each value in the column, relative to the
               column mean and standard deviation
             * Then it takes the absolute of Z-score because the direction does not matter, only if it is
               below the threshold.
             * all(axis=1) ensures that for each row, all column satisfy the constraint.
             * Finally, result of this condition is used to index the data frame.
        """
        self.df = self.df[(np.abs(stats.zscore(self.df)) < 3).all(axis=1)]

    def __remove_unknown_values(self):
        """
            Function replace 0 values in some columns to mean value of the column
        """
        # turn zero values into NaN values
        self.df = self.df.mask(self.df == 0)
        # calculate all feature means
        means = self.df.mean()
        # replace NaN values with mean values
        self.df = self.df.fillna(means)

    def __transform_categorical_data(self):
        """
            Function to replace categorical column values into numerical
        """
        lbl_encoder = LabelEncoder()
        self.df = self.df[self.df.columns[:]].apply(lbl_encoder.fit_transform)

    def __normalize(self):
        """
            Function to normalize feature values with standard deviation
        """
        # create standard scaler
        sc = StandardScaler()
        # transform feature values to normalized
        self.features_ = sc.fit_transform(self.features_)

def main():
    """ Initialise DataFrame and pull the features and targets """
    df = DataFeeder()
    features, target = df.get_data()
    
    """ Use only 1 component """
    features = df.pca(n_components=1)
    
    """ Split features and target into 70% train and 30% test """
    features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=0.3, stratify=target, random_state=100)
    
    """ Initialise Gaussian Naive Bayes into variable clf """
    clf = GaussianNB()
    
    """ Fit the training data into the classifier and predict using test data """
    
    y_pred = clf.fit(features_train, target_train).predict(features_test)
    
    """ Calculate and print accuracy score """
    acc = accuracy_score(target_test, y_pred) * 100
    print(acc)

if __name__ == '__main__':
    main()
