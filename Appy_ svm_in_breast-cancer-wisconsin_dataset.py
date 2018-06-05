import numpy as np
from sklearn import cross_validation, svm
import pandas as pd

df = pd.read_csv('data/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)


X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])


# split train and test data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


# fit our data
clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print("The model Accuracy is ", accuracy)