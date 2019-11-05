import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier        #Decision Tree
from sklearn.ensemble import RandomForestClassifier    #Random Forest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
import matplotlib.pylab as plt
from matplotlib import cm
import seaborn as sns
import numpy as np



#load the breast cancer data
cancer = pd.read_csv('cancer.csv')
X = cancer.iloc[:, 1:31].values
Y = cancer.iloc[:, 31].values

cancer.head()

print("#########################################################")
print("Cancer data set dimensions : {}".format(cancer.shape))
print("#########################################################")
print("---------------------------------------------------------")
      
cancer.groupby('diagnosis').size()

#view the data
print(cancer.describe)
print(cancer.columns)
print(cancer.diagnosis)
print(cancer.shape)


#Visualization of data by grouping malignant and benign tumours
cancer.groupby('diagnosis').hist(figsize=(12, 12))
cancer.plot.scatter(x='radius_mean', y='concavity_mean', c='blue',colormap = cm.get_cmap('Spectral'))

cancer.isnull().sum()
cancer.isna().sum()

dataframe = pd.DataFrame(Y)

#Encoding categorical data values 
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(Y)
"""
#changing 'M' and 'B' labels to 1s and 0s\n",
dataset['diagnosis'].replace('M', 1.0,inplace=True)
dataset['diagnosis'].replace('B', 0.0,inplace=True)
"""

#----------------- Decision Tree
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

neighbors_setting = range(1,15)

training_accuracy = []
test_accuracy = []

max_dep = range(1,15)

for md in max_dep:
    tree = DecisionTreeClassifier(max_depth=md,random_state=0)
    tree.fit(X_train,y_train)
    training_accuracy.append(tree.score(X_train, y_train))
    test_accuracy.append(tree.score(X_test, y_test))
 
plt.plot(max_dep,training_accuracy, label='Accuracy of the training set')
plt.plot(neighbors_setting,test_accuracy, label='Accuracy of the test set')
plt.ylabel('Accuracy')
plt.xlabel('Max Depth')
plt.legend()

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# By having larger max_depth (>5), we overfit the model into training data, so the accuracy for training set increases 
# but the accuracy for test set decreases

# other parameters than we can work with:
# - min_samples_leaf, max_sample_leaf
# - max_leaf_node

# by looking at plot, best result accurs when max_depth is 3
#----------------- Decision Tree


#Export Tree
export_graphviz(tree, out_file='cancerTree.dot', class_names=['malignant','benign'], feature_names=cancer.feature_names, impurity=False, filled=True)

print('Feature importances: {}'.format(tree.feature_importances_))
type(tree.feature_importances_)

#Feature Importance
n_feature = cancer.data.shape[1]
plt.barh(range(n_feature), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_feature), cancer.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

#predicting the Test set results
Y_pred = classifier.predict(X_test)

#Creating the confusion Matrix
cm = confusion_matrix(y_test, Y_pred)
sns.heatmap(cm, annot=True)
c = print(cm[0, 0] + cm[1, 1])
plt.savefig('E:/University/Artificial Intelligence/confusion_matrix.png')