import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, classification_report
from scikitplot.metrics import plot_confusion_matrix
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
from subprocess import check_call


from data_feeder import DataFeeder


def find_best_params(features, target, n_folds=5, scorer=None):
    """
      Function to find best hyperparameters for Decision Tree Classifier

      :param features - Dataset feature columns
      :param target - Dataset target column
      :param n_folds - Number fold for GridSearchCV
      :param scorer - Scorer for GridSearchCV
    """
    params = {'max_depth': np.linspace(1, 32, 32, endpoint=True)}
    __search = GridSearchCV(DecisionTreeClassifier(),
                            params, cv=n_folds, scoring=scorer)
    __search.fit(features, target)
    return __search.best_params_


def std_train_test_split(features_train, features_test, target_train, target_test, max_depth=5):
    """
      Function to train and test Decision Tree Classifier
      :param features_train - Dataset feature columns for training
      :param feataures_test - Dataset feature columns for testing
      :param target_train - Dataset target column for training
      :param target_test - Dataset target column for testing
      :param max_depth - Maximum depth of the classifier
    """
    print('############  Standard Train/Test Split  #################')
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(features_train, target_train)
    target_pred = clf.predict(features_test)

    acc = accuracy_score(target_test, target_pred) * 100
    print('Accuracy Score: %.2f' % acc)
    # classification report
    print(classification_report(target_test,
                                target_pred, target_names=["B", "M"]))
    # plot confusion matrix
    plot_confusion_matrix(target_test, target_pred)
    # export graph
    with open('decision_tree.dot', 'w') as f:
        f = export_graphviz(clf, out_file=f)
    check_call(['dot', '-Tpng', 'decision_tree.dot',
                '-o', 'decision_tree.png'])


def cross_validation(features, target, max_depth=5, n_splits=5, random_state=1):
    """
      Function to perform cross validation of the classifier
      :param features - Dataset feature columns
      :param target - Dataset target column
      :param n_splits - Number of cross validation splits
      :param random_state - Number for random state generator
    """
    print('############   Running Cross Validation   ######################')
    kf = StratifiedKFold(
        n_splits=n_splits, random_state=random_state).split(features, target)
    clf = DecisionTreeClassifier(max_depth=max_depth)
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

    acc = np.mean(scores)
    error_rate = np.std(scores)
    print('Accuracy Score: %.2f (+- %.2f)' % (acc, error_rate))
    print('F1 Score: %.2f' % (s_f1 / n_splits))
    print('Precision Score: %.2f' % (s_prec / n_splits))
    print('Recall Score: %.2f' % (s_rec / n_splits))


def main():
    # initialize data feeder
    df = DataFeeder()
    # get feature and target data sets
    features, target = df.get_data()

    # perform PCA
    features = df.pca(n_components=4)
    # features = df.pca(n_components=10)

    # find best hyperparameters
    scorer = make_scorer(f1_score, pos_label=0)
    params = find_best_params(features, target, scorer=scorer)

    features_train, features_test, target_train, target_test = train_test_split(
        features, target, stratify=target, random_state=1)

    # run training and testing data split
    std_train_test_split(features_train, features_test,
                         target_train, target_test, max_depth=int(params['max_depth']))

    # run cross validation
    cross_validation(features, target, max_depth=int(params['max_depth']))
    plt.show()


if __name__ == '__main__':
    main()
