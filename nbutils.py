import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('fma_metadata/cleaned_dataset_headers.csv')

X = dataset.iloc[:, 1:519]
y = dataset.iloc[:, 519]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.1)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))   

# PRINT TIME AFTER EXECUTION
import time
ts = time.time()
print ts
import datetime
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print st