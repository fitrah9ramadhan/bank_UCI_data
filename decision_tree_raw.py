import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

# set_option pandas
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# load data
df = pd.read_csv('data/bank-full.csv', delimiter=";")

# Preprocessing
df = pd.get_dummies(columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'], data=df)
df['y'] = df['y'].map({'yes':1, 'no':0})


# Feature and target
X = df.drop(columns='y')
y = df['y']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Decision Tree Model
clf = DecisionTreeClassifier()
tree_model = clf.fit(X_train, y_train)

# Model Evaluation
y_pred = tree_model.predict(X_test)

acc_score = round(accuracy_score(y_pred, y_test), 3)

print('Accuracy without cross validation: ', acc_score)

# Using Cross Validation
scores = cross_val_score(clf, X, y, cv=5)
print("Accuracy with 5 Fold Cross Validation", scores)
print("Average Accuracy: ", scores.mean())
