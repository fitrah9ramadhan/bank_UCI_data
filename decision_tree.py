import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
# set_option pandas
# pd.set_option('display.max_columns', 50)
# pd.set_option('display.width', 1000)

# load data
df = pd.read_csv('data/cleaned_data/bank-full-ready.csv', index_col='Unnamed: 0')

# Feature and target
X = df.drop(columns='y')
y = df['y']

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Decision Tree Model
clf = DecisionTreeClassifier()

# model and prediction
model = clf.fit(X_train, y_train)
preds = model.predict(X_test)

classification_report = classification_report(preds, y_test)

print(f"Classification Report: \n {classification_report}")