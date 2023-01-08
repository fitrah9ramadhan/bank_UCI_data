import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

accuracy = accuracy_score(preds, y_test)
precision = precision_score(preds, y_test)
recall = recall_score(preds, y_test)
f1_score = f1_score(preds, y_test)

dct = {'accuracy': [round(accuracy, 2)], 
	   'precision': [round(precision, 2)], 
	   'recall': [round(recall, 2)], 
	   'f1_score': [round(f1_score, 2)]}
metrics = pd.DataFrame(data=dct)

print("=== Imbalance datasets ===")
print(metrics)