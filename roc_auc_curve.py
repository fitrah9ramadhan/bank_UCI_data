import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt

# load data
df = pd.read_csv('data/cleaned_data/bank-full-balanced.csv', index_col='Unnamed: 0')

# Feature and Target
X = df.drop(columns=['y'], axis=1)
y = df['y']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# logistic regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_probs = lr_model.predict_proba(X_test)
lr_probs = lr_probs[:, 1]

# Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_probs = rf_model.predict_proba(X_test)
rf_probs = rf_probs[:, 1]

# Gaussian Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_probs = nb_model.predict_proba(X_test)
nb_probs = nb_probs[:, 1]

# Calculating AUC Score
lr_auc = roc_auc_score(y_test, lr_probs)
rf_auc = roc_auc_score(y_test, rf_probs)
nb_auc = roc_auc_score(y_test, nb_probs)

# Generating ROC
# fpr: False Positive Rate
# tpr: True Positive Rate
lr_fpr, lr_tpr, threshold = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, threshold = roc_curve(y_test, rf_probs)
nb_fpr, nb_tpr, threshold = roc_curve(y_test, nb_probs)


plt.subplots(figsize=(10, 10))

plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC: {(round(lr_auc, 3))})")
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest Classifier (AUC: {(round(rf_auc, 3))})")
plt.plot(nb_fpr, nb_tpr, label=f"Gaussian Navie Bayes Classifier (AUC: {(round(nb_auc, 3))})")

plt.legend()

plt.suptitle("ROC Curve")
plt.title("Logistic Regression Vs. Random Forest Vs. Gaussian Naive Bayes")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.savefig("figure/ROC_AUC_Score.png")
plt.show()