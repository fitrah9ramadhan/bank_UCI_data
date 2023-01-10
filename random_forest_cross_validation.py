import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# load data
df = pd.read_csv('data/cleaned_data/bank-full-balanced.csv', index_col='Unnamed: 0')

# Feature and target
X = df.drop(columns='y')
y = df['y']

# Random Forest
rf_model = RandomForestClassifier()

# Cross Validation
metrics = ['accuracy', 'precision', 'recall', 'f1']

for metric in metrics:

	print(f"Metrics: {metric}")
	cv = cross_val_score(rf_model, X, y, cv=10, scoring=metric)
	print(cv)
