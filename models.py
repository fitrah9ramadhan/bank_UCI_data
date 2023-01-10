import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

#set_option pandas
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# load data
df = pd.read_csv('data/cleaned_data/bank-full-balanced.csv', index_col='Unnamed: 0')

# Feature and target
X = df.drop(columns='y')
y = df['y']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


models = {
	'Logistic Regression' : LogisticRegression(max_iter=1000),
	'Decision Tree Classifier' : DecisionTreeClassifier(),
	'Random Forest Classifier' : RandomForestClassifier(),
	'Gaussian Naive Bayes' : GaussianNB()
}

def print_score(model, X_train, X_test, y_train, y_test, model_name):

	my_model = model
	my_model.fit(X_train, y_train)
	my_preds = my_model.predict(X_test)

	my_accuracy = accuracy_score(y_test, my_preds)
	my_precision = precision_score(y_test, my_preds)
	my_recall = recall_score(y_test, my_preds)
	my_f1_score = f1_score(y_test, my_preds)

	report = classification_report(y_test, my_preds)

	print(f"{model_name} : accuracy : {round(my_accuracy, 2)}")
	print(f"{model_name} : precision : {round(my_precision, 2)}")
	print(f"{model_name} : recall : {round(my_recall, 2)}")
	print(f"{model_name} : f1 score : {round(my_f1_score, 2)}")
	print("=============================================")
	print(f"Model Name: {model_name}")
	print(report)
	print("=============================================")


for model_name, model in models.items():
	print_score(model, X_train, X_test, y_train, y_test, model_name)