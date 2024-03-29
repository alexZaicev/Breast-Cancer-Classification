import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scikitplot.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data_feeder import DataFeeder


def plot_decision_regions(X, y, classifier, resolution=.02, test_idx=None):
    """
        Function to print decision boundaries
    """
    # setup marker generator & color map
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


def run_pca(X, n_components=2, columns=('pc_1', 'pc_2')):
    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(X)
    return pd.DataFrame(data=pc, columns=columns)


def calculate_f1_score(y_test, y_pred):
    """
        Function calculate precision, recall and F1-score of your model
    :param y_test: Testing data set
    :param y_pred: Actually predicted data set
    """
    print('# Running precision, recall and F1-score')
    print('# F1-Score:\t\t%.2f' % f1_score(y_test, y_pred, average="macro"))
    print('# Precision:\t%.2f' % precision_score(y_test, y_pred, average="macro"))
    print('# Recall:\t\t%.2f' % recall_score(y_test, y_pred, average="macro"))

def run_2d_model():
    """
        2D example
    """
    print('\nLinear Discriminant Analysis - 2 dimensions with decision regions\n')
    # get features of the data and the target
    dt = DataFeeder()
    X, y = dt.get_data()
    # reduce our features only to 2 dimensions
    X = run_pca(X)
    # split data into 70% training & 30% testing
    X_train_std, X_test_std, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    # create linear dicriminant analysis model
    model = LinearDiscriminantAnalysis()
    # train
    model.fit(X_train_std, y_train)
    # test
    y_pred = model.predict(X_test_std)
    # calculate model accuracy score
    score = accuracy_score(y_test, y_pred) * 100
    print('# Accuracy score: %.2f' % score)
    calculate_f1_score(y_test, y_pred)

    # prepare data for visualization
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined_std = np.hstack((y_train, y_test))
    # plot decision boundaries
    plt.figure()
    plot_decision_regions(X_combined_std, y_combined_std, model)
    # plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, normalize=True, title='Confusion Matrix')
    plt.show()


def run_4d_model():
    """
        4D example
    """
    print('\nLinear Discriminant Analysis - 4 dimensions\n')
    # get features of the data and the target
    dt = DataFeeder()
    X, y = dt.get_data()
    # reduce our features only to 2 dimensions
    X = run_pca(X, n_components=4, columns=['pc_1', 'pc_2', 'pc_3', 'pc_4'])
    # split data into 70% training & 30% testing
    X_train_std, X_test_std, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    # create linear dicriminant analysis model
    model = LinearDiscriminantAnalysis()
    # train
    model.fit(X_train_std, y_train)
    # test
    y_pred = model.predict(X_test_std)
    # calculate model accuracy score
    score = accuracy_score(y_test, y_pred) * 100
    print('# Accuracy score: %.2f' % score)
    calculate_f1_score(y_test, y_pred)
    # plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, normalize=True, title='Confusion Matrix')
    plt.show()

# run_2d_model()
run_4d_model()
