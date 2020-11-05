# Importing the libraries
import numpy as np
import pandas as pd
import pickle
 
# Importing the dataset
dataset = pd.read_csv('trainingUNSW.csv')

# Encoding the attack_cat column (categorical data)
df = pd.get_dummies(dataset, columns=['attack_cat'])

# Drop the null values in df
df = df.dropna()

# Store the 'label' values in y and y_test
y = df['label'].values

# Dropping the label column
df = df.drop(['label'], axis=1)

# Selecting the first 40 features
X = df.iloc[:, :40].values

# Feature scaling to normalize data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Save the Scalar
pickle.dump(sc, open('scaler.pkl', 'wb'))

# Fetching the classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
classifier = RandomForestClassifier()
classifier.fit(X, y)

# Save the model to disk
filename = 'RF.sav'
print("first line")
pickle.dump(classifier, open(filename, 'wb'))
print("done")
