# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from data_feeder import DataFeeder

def plot_decision_regions(X, y, classifier, resolution=.02, test_idx=None):
    """
        Plot decision boundaries
    """
    plt.figure()
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

def find_parameters(features, target, n_folds=5, scorer=None, penalty='l2', solver='lbfgs'):
    """
        Find best hyperparameters for Logistic regression
    """
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    clf = GridSearchCV(LogisticRegression(penalty), param_grid=params)
    clf.fit(features, target)
    return clf.best_params_


def run_train_test_split(features, target, random_state=1, penalty='l2', solver='lbfgs', C=1, title=''):
    """
        Run training and testing dataset split model
    """
    X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target, random_state=random_state)
    clf = LogisticRegression(penalty=penalty, solver=solver)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    acc = accuracy_score(y_test, y_predict) * 100
    
    print('Accuracy Score: %.2f' % acc)
    # plot confusion matrix
    plot_confusion_matrix(y_test, y_predict)
    # plot boundaries
    if len(features.columns) == 2:
        X = np.vstack((X_train, X_test))
        y = np.hstack((y_train, y_test))
        plot_decision_regions(X, y, clf, test_idx=range(len(X_train) - 1, len(features) - 1))
    

def run_cross_validation(features, target, C=1, random_state=1, n_splits=5, penalty='l2', solver='lbfgs', title=''):
    """
        Run cross validation model
    """
    kf = StratifiedKFold(
        n_splits=n_splits, random_state=random_state).split(features, target)
    clf = LogisticRegression(C=C, penalty=penalty, solver=solver)
    s_f1, s_prec, s_rec = 0, 0, 0
    scores = []
    for k, (train, test) in enumerate(kf):
        clf.fit(features.iloc[train], target.iloc[train])
        # predict and get accuracy score
        target_pred = clf.predict(features.iloc[test])
        scores.append(accuracy_score(target.iloc[test], target_pred) * 100)

        s_f1 += f1_score(target.iloc[test],
                         target_pred.round(), average="macro") * 100
        s_prec += precision_score(target.iloc[test],
                                  target_pred.round(), average="macro") * 100
        s_rec += recall_score(target.iloc[test],
                              target_pred.round(), average="macro") * 100
    plt.title(title)
    acc = np.mean(scores)
    error_rate = np.std(scores)
    print('Accuracy Score: %.2f (+- %.2f)' % (acc, error_rate))
    print('F1 Score: %.2f' % (s_f1 / n_splits))
    print('Precision Score: %.2f' % (s_prec / n_splits))
    print('Recall Score: %.2f' % (s_rec / n_splits))


def main():
    # create data feeder and get features and target
    dt = DataFeeder()
    features, target = dt.get_data()
    
    # perform PCA with variety of components
    # features = dt.pca(2)
    features = dt.pca(10)
    
    # get best hyperparameters
    scorer = make_scorer(f1_score, pos_label=0)
    params = find_parameters(features, target, scorer=scorer)
    
    # run train test split without penalty
    print('#################################################')
    print('Train test split without penaty')          
    run_train_test_split(features, target, C=params['C'], penalty='none', solver='saga')
    # run train test split with L2 penalty
    print('#################################################')
    print('Train test split with L2 penaty')
    run_train_test_split(features, target, C=params['C'])
    # run cross validation with L2 penalty
    print('#################################################')
    print('Cross Validation with L2 penalty')
    run_cross_validation(features, target, C=params['C'], penalty='none', solver='saga', title='Cross validation with no penalty')
    # run cross validation without penalty
    print('#################################################')
    print('Cross Validation without penalty')
    run_cross_validation(features, target, C=params['C'], title='Cross validation with l2 penalty')
    
    # plot decission boundaries
    plt.show()
    

if __name__ == '__main__':
    main()