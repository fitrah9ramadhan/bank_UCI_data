import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

# set_option pandas
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# load data
df = pd.read_csv('data/cleaned_data/bank-full-ready.csv', index_col='Unnamed: 0')
print(df.info())
# Feature and target
X = df.drop(columns='y')
y = df['y']


# Decision Tree Model
clf = DecisionTreeClassifier()

# Using Cross Validation
scores = cross_val_score(clf, X, y, cv=10)

for (i, score) in enumerate(scores, start=1):
	print('Fold - {}: {}'.format(i, round(score, 2)))
print("Average Accuracy: ", round(scores.mean(), 2))