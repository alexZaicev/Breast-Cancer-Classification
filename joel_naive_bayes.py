""" Artifical Intelligence and Machine Learning Coursework """
""" Naive Bayes prototype """

""" CODE CURRENTLY NOT WORKING """

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

# Load the data

df = pd.read_csv('D:\BREAST_CANCER_WISCONSIN.csv')

# Edit the DataFrame to only contain 2 columns

df = df [["diagnosis",
          "radius_mean"]]

# Assign the number 0 to letter M, and 1 to letter B

df["label"] = df["diagnosis"].apply(lambda x: 0 if x=="M" else 1)

# Split the data into Training and Test sets
# Splitting it into 70% training and 30% test

X_train, X_test, y_train, y_test = train_test_split(df.radius_mean, df.label, test_size=0.3, random_state=100)
# X_train, X_test, y_train, y_test = train_test_split(df["radius_mean"], df["label"], test_size=0.3, random_state=100)

# Without below reshape, receive below error:
# ValueError: Expected 2D array, got 1D array instead:
# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

X_train.values.reshape(-1, 1)
y_train.values.reshape(-1, 1)

# Initiate the Gaussian Naive Bayes classifer

clf = GaussianNB()
clf.fit(X_train, y_train)