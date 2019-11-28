""" Artifical Intelligence and Machine Learning Coursework """
""" Naive Bayes prototype """

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.naive_bayes import GaussianNB

from data_feeder import DataFeeder

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
    print("Accuracy Score: %.2f" % acc)
    print("F1 score: %.2f" % (f1_score(target_test, y_pred) * 100))
    print("Recall score: %.2f" % (recall_score(target_test, y_pred) * 100))
    print("Precision score: %.2f" % (precision_score(target_test, y_pred) * 100))

if __name__ == '__main__':
    main()