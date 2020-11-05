# Importing the libraries
import numpy as np
import pandas as pd
import pickle

df_test = pd.read_csv('testingUNSW.csv')
df_test = pd.get_dummies(df_test, columns=['attack_cat'])
y_test = df_test['label'].values
df_test = df_test.drop(['label'], axis=1)
X_test = df_test.iloc[:, :40].values

from sklearn.preprocessing import StandardScaler
sc = pickle.load(open('scaler.pkl', 'rb'))
X_test = sc.transform(X_test)

# Load the model from disk
filename = 'RF.sav'
clf = pickle.load(open(filename, 'rb'))

from sklearn.metrics import accuracy_score
train_predictions = clf.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))
