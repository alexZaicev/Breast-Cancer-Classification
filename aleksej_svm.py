"""

    Created Date: 2019-10-15
    @author Aleksej Zaicev
    Student ID: S15125327

"""

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    make_scorer,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.svm import SVC, LinearSVC

from data_feeder import DataFeeder

# disable unwanted future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


class Plotter(object):
    """
        Class to handle plotting data into various charts
    """

    @staticmethod
    def transform_data_sets(feature_train, feature_test, target_train, target_test):
        """
            Function to combine features and target data set into and NumPy array depending on the stack
        :param feature_train: Features data set for training
        :param feature_test: Features data set for testing
        :param target_train: Target data set for training
        :param target_test: Target data set for testing
        :return: Combined features and target data sets
        """
        f_combined = np.vstack((feature_train, feature_test))
        t_combined = np.hstack((target_train, target_test))
        return f_combined, t_combined

    @staticmethod
    def plot_decision_regions(features, target, clf, test_idx=None, title=None):
        """
            Function to plot decision regions
        :param features: Features data set
        :param target: Target data set
        :param clf: Classifier model
        :param test_idx: Indexes of test data to circle on the plot
        :param title: Title of the plot figure
        """
        plt.figure()
        if title is not None:
            plt.title(title)
        # setup marker generator and color map
        markers = ("s", "x", "o", "^", "v")
        colors = ("red", "blue", "lightgreen", "gray", "cyan")
        cmap = ListedColormap(colors[: len(np.unique(target))])

        # plot the decision surface
        x1_min, x1_max = features[:, 0].min() - 1, features[:, 0].max() + 1
        x2_min, x2_max = features[:, 1].min() - 1, features[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02)
        )
        z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        z = z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(target)):
            plt.scatter(
                x=features[target == cl, 0],
                y=features[target == cl, 1],
                alpha=0.8,
                c=colors[idx],
                marker=markers[idx],
                label=cl,
                edgecolor="black",
            )

        # highlight test samples
        if test_idx:
            # plot all samples
            features_test, target_test = features[test_idx, :], target[test_idx]
            plt.scatter(
                features_test[:, 0],
                features_test[:, 1],
                c="",
                edgecolor="black",
                alpha=1.0,
                linewidth=1,
                marker="o",
                s=100,
                label="test set",
            )

    @staticmethod
    def plot_confusion_matrix(target_test, target_pred, normalize=True, title=None):
        """
            Function to plot confusion matrix
        :param target_test: Target data set for testing
        :param target_pred: Predicted target data set
        :param normalize: Normalize values
        :param title: Title of the plot figure
        """
        plot_confusion_matrix(
            target_test, target_pred, normalize=normalize, title=title
        )

    @staticmethod
    def plot_roc(all_roc_auc, all_fpr, all_tpr, mean_tpr, mean_fpr, title=None):
        """
            Function to plot receiver operating characteristics
        :param all_roc_auc: All ROC areas under the curve
        :param all_fpr: All false positive rates
        :param all_tpr: All true positive rates
        :param mean_tpr: Mean of true positive rates
        :param mean_fpr: Mean of false positive rates
        :param title: Title of the plot figure
        """
        plt.figure()
        if title is not None:
            plt.title(title)
        for i in range(0, len(all_tpr), 1):
            plt.plot(
                all_fpr[i],
                all_tpr[i],
                label="ROC fold %d (area = %.2f" % (i + 1, all_roc_auc[i]),
            )
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color=(0.6, 0.6, 0.6),
            label="random guessing",
        )
        mean_tpr /= len(all_tpr)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(
            mean_fpr, mean_tpr, "k--", label="mean ROC (area = %.2f)" % mean_auc, lw=2
        )
        plt.plot(
            [0, 0, 1],
            [0, 1, 1],
            linestyle=":",
            color="black",
            label="perfect performance",
        )
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.legend(loc="lower right")

    @staticmethod
    def plot_distribution(features, xlim=None, title='', ylabel='', xlabel='', bins=None):
        """
            Function to plot distribution histograb
        """
        plt.figure()
        plt.hist(features, edgecolor="black", bins=bins)
        # Add labels
        if xlim is not None:
            axes = plt.axes()
            axes.set_xlim(xlim)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)


class Evaluator(object):
    """
        Class holding machine learning model evaluation methods
    """

    def __init__(self):
        self.clf_ = None

    def predict(
        self, features_train, target_train, features_test, target_test, title=None
    ):
        """
            Function to run classifier prediction and plot metric graphs
        :param features_train: Features data set for training
        :param target_train: Target data set for training
        :param features_test: Features data set for test
        :param target_test: Target data set for test
        :param title: Title of the plot figure
        :return:
        """
        if self.clf_ is None:
            raise AttributeError("Classifier have not been initialized")
        target_pred = self.clf_.predict(features_test)
        score = Evaluator.accuracy(target_test, target_pred)
        print("ML model accuracy score: %.2f" % score)

        # print out classification report
        print(classification_report(target_test, target_pred, target_names=["B", "M"]))

        if len(features_train.columns) == 2:
            # plot decision boundaries
            fc, tc = Plotter.transform_data_sets(
                features_train, features_test, target_train, target_test
            )
            Plotter.plot_decision_regions(
                fc,
                tc,
                clf=self.clf_,
                test_idx=range(len(target_train), len(tc)),
                title=title,
            )
        # plot confusion matrix
        Plotter.plot_confusion_matrix(target_test, target_pred, title=title)

    def k_fold_cv(
        self,
        features,
        target,
        n_splits=10,
        random_state=1,
        linear_params=None,
        rbf_params=None,
    ):
        """
            Function to perform stratified K-fold cross validation training and testing.
            Receiver operating characteristics plot will be generated to analyze how well
            model can predict
        :param features: Features data set
        :param target: Target data set
        :param n_splits: Number of K-fold splits
        :param random_state: Random state for random data set pick
        :param linear_params: Hyper-parameters for Linear SVM classifier
        :param rbf_params: Hyper-parameters for Gaussian SVM classifier
        """
        if linear_params is None:
            linear_params = {"C": 1}
        if rbf_params is None:
            rbf_params = {"C": 1, "gamma": 0.1}

        # create 2 classifiers with different SVM kernels
        __clfs = [
            SVC(kernel="linear", C=linear_params["C"], probability=True),
            SVC(
                kernel="rbf",
                C=rbf_params["C"],
                gamma=rbf_params["gamma"],
                probability=True,
            ),
        ]

        for __clf in __clfs:
            kf = StratifiedKFold(n_splits=n_splits, random_state=random_state).split(
                features, target
            )

            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            all_fpr, all_tpr, all_roc_auc = list(), list(), list()

            if __clf.kernel == "linear":
                name = "Linear"
            else:
                name = "Gaussian"

            print("#########################################################")
            print(
                "Running Stratified K-fold Cross Validation on {} kernel SVM".format(
                    name
                )
            )
            scores = list()
            for k, (train, test) in enumerate(kf):
                __clf.fit(features.iloc[train], target.iloc[train])
                # predict and get accuracy score
                target_pred = __clf.predict(features.iloc[test])
                scores.append(self.accuracy(target.iloc[test], target_pred))
                # get prediction probability
                probas = __clf.predict_proba(features.iloc[test])
                # calculate false/true positive rate and thresholds
                fpr, tpr, thresholds = roc_curve(
                    target[test], probas[:, 1], pos_label=1
                )
                mean_tpr += np.interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                # get area under the curve
                all_roc_auc.append(auc(fpr, tpr))
                all_fpr.append(fpr)
                all_tpr.append(tpr)
            print(
                "Finished Stratified K-fold Cross Validation on {} kernel SVM".format(
                    name
                )
            )
            cv_mean = np.mean(scores)
            cv_std = np.std(scores)
            print("CV accuracy: %.2f (+/- %.2f)" % (cv_mean, cv_std))
            Plotter.plot_roc(
                all_roc_auc,
                all_fpr,
                all_tpr,
                mean_tpr,
                mean_fpr,
                title="{} SVM".format(name),
            )

    @staticmethod
    def split(
        features, target, stratify=None, random_state=1, test_size=0.3, shuffle=True
    ):
        """
            Function to split features and target data sets into training and testing pieces
            that will be fed into ML model
        :param features: Features data set
        :param target: Target data set
        :param stratify: Data set to include proportional data split
        :param random_state: Random state for picking random sets
        :param test_size: Proportion of testing samples
        :param shuffle: Shuffle data before splitting
        :return: features train & test, target train & test
        """
        return train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
            shuffle=shuffle,
        )

    @staticmethod
    def accuracy(target_test, target_pred, percentage=True):
        """
            Function to calculate accuracy score value
        :param target_test: Target testing data set
        :param target_pred: Target predicted data set
        :param percentage: Return accuracy as percentage
        :return: accuracy
        """
        score = accuracy_score(target_test, target_pred)
        if percentage:
            score *= 100
        return score

    @staticmethod
    def find_best_params(features_train, target_train, n_folds=5, scoring=None):
        """
            Function to find best hyper-parameter for SVM using GridSearchCV
        :param features_train: Features data set for training
        :param target_train: Target data set for training
        :param n_folds: Number of CV folds
        :param scoring: Scoring system
        :return: parameters for linear and Gaussian kernel SVMs
        """
        __linear_param_bucket = {"C": [1, 10, 100, 1000]}
        __rbf_param_bucket = {
            "C": [1, 10, 100, 1000],
            "gamma": [0.1, 0.01, 0.001, 0.0001],
        }

        # find linear SVM params
        __search = GridSearchCV(
            SVC(kernel="linear"), __linear_param_bucket, cv=n_folds, scoring=scoring
        )
        __search.fit(features_train, target_train)
        linear_prams_ = __search.best_params_
        # find Gaussian SVM params
        __search = GridSearchCV(
            SVC(kernel="rbf"), __rbf_param_bucket, cv=n_folds, scoring=scoring
        )
        __search.fit(features_train, target_train)
        rbf_prams_ = __search.best_params_
        return linear_prams_, rbf_prams_

    def run_linear_svm(
        self, features_train, features_test, target_train, target_test, params=None
    ):
        """
            Function to train and test Linear kernel SVM
        :param features_train: Features data set for training
        :param features_test: Features data set for testing
        :param target_train: Target data set for training
        :param target_test: Target data set for testing
        :param params: SVM hyper-parameters
        """
        print(
            "########################################################################"
        )
        print("Running Linear Kernel Support Vector Machines Model....")
        if params is None:
            params = {"C": 1}
        print("SVM hyper-parameters: C={}".format(params["C"]))
        self.clf_ = LinearSVC(
            C=params["C"], loss="squared_hinge", penalty="l2", tol=0.0001
        )
        self.clf_.fit(features_train, target_train)
        print("Finished Linear Kernel Support Vector Machines Model....")
        self.predict(
            features_train, target_train, features_test, target_test, title="Linear SVM"
        )

    def run_rbf_svm(
        self, features_train, features_test, target_train, target_test, params=None
    ):
        """
            Function to train and test Gaussian kernel SVM
        :param features_train: Features data set for training
        :param features_test: Features data set for testing
        :param target_train: Target data set for training
        :param target_test: Target data set for testing
        :param params: SVM hyper-parameters
        """
        print(
            "########################################################################"
        )
        print("Running Gaussian Kernel Support Vector Machines Model....")
        if params is None:
            params = {"C": 1, "gamma": 0.1}
        print(
            "SVM hyper-parameters: C={}; gamma={}".format(params["C"], params["gamma"])
        )
        self.clf_ = SVC(kernel="rbf", C=params["C"], gamma=params["gamma"])
        self.clf_.fit(features_train, target_train)
        print("Finished Gaussian Kernel Support Vector Machines Model....")
        self.predict(
            features_train,
            target_train,
            features_test,
            target_test,
            title="Gaussian SVM",
        )


def main():
    """
        Main function containing object initialization and method triggering order
    """
    # data feeding object
    df = DataFeeder()
    # evaluation object
    ev = Evaluator()
    # get features and target data sets
    features, target = df.get_data(normalize=False)

    Plotter.plot_distribution(target, ["M", "B"], bins=2,title="Diagnosis Distribution", xlabel="Diagnosis", ylabel="Records")
    Plotter.plot_distribution(features.iloc[:, 1], bins=50, title="Texture Mean Distribution", xlabel="Texture Mean", ylabel="Records")
    Plotter.plot_distribution(features.iloc[:, 2],bins=50, title="Perimeter Mean Distribution", xlabel="Perimeter Mean", ylabel="Records")
    # get features and target data sets
    features, target = df.get_data()

    # run PCA
    # features = df.pca(n_components=2)
    # features = df.pca(n_components=4)
    features = df.pca(n_components=10)

    # split data
    features_train, features_test, target_train, target_test = Evaluator.split(features, target, stratify=target)
    # find best parameters based on F1-score
    scorer = make_scorer(f1_score, pos_label=0)
    linear_params, rbf_params = Evaluator.find_best_params(features_train, target_train, n_folds=10, scoring=scorer)
    # train and test model trained on K-fold cross validation
    ev.k_fold_cv(features, target, n_splits=10, linear_params=linear_params, rbf_params=rbf_params)
    # train and test linear SVM model with best parameter
    ev.run_linear_svm(features_train, features_test, target_train, target_test, params=linear_params)
    # train and test rbf SVM model with best parameter
    ev.run_rbf_svm(features_train, features_test, target_train, target_test, params=rbf_params)
    # show all plot figures
    plt.show()


# statement indicating main file
if __name__ == "__main__":
    main()
