from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MLModel(ABC):
    """
        Machine learning model base class
    """

    @abstractmethod
    def fit(self, X, y):
        """ Fit training data
            :param X - training vectors
            :param y - target values
            :returns self object
        """
        return self

    @abstractmethod
    def predict(self, X):
        """ Fit training data
            :param X - testing vectors
            :returns target prediction
        """
        return list()


def main():
    data = import_data()
    # TODO: pre-process data
    # data features
    X = data.iloc[:, 2:]
    # target to predict
    y = data.iloc[:, 1]
    # split dataset into training & testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    # initialize ML model
    model = MLModel()
    # start training
    model.fit(X_train, y_train)
    # predict
    y_pred = model.predict(X_test)
    # print accuracy score
    print(accuracy_score(y_test, y_pred, normalize=True) * 100)


def import_data():
    return pd.read_csv('datasets\\BREAST_CANCER_WISCONSIN.csv', header=0)


if __name__ == '__main__':
    main()
