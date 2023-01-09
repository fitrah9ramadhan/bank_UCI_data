import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix

#set_option pandas
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# load data
df = pd.read_csv('data/cleaned_data/bank-full-balanced.csv', index_col='Unnamed: 0')

# Feature and target
X = df.drop(columns='y')
y = df['y']


# Metrics
metrics = ['accuracy', 'precision', 'recall', 'f1']

# Logistic Regression
clf = LogisticRegression(random_state=1, max_iter=1000)


eval_dct_lr = dict()
for metric in metrics:
	cv = cross_val_score(clf, X, y, cv=5, scoring=metric)
	eval_dct_lr[metric]=cv

model_evaluation_lr = pd.DataFrame(eval_dct_lr)
print(model_evaluation_lr)
print(model_evaluation_lr.mean())

# Decision Tree Model
clf = DecisionTreeClassifier(random_state=1)

eval_dct_dt = dict()
for metric in metrics:
	cv = cross_val_score(clf, X, y, cv=5, scoring=metric)
	eval_dct_dt[metric]=cv

model_evaluation_dct = pd.DataFrame(eval_dct_dt)

print(model_evaluation_dct)
print(model_evaluation_dct.mean())


