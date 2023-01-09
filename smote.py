import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

df = pd.read_csv('data/cleaned_data/bank-full-ready.csv', index_col='Unnamed: 0')

X = df.drop(columns=['y'], axis=1)
y = df['y']

smote = SMOTE(random_state=1)
X_smote, y_smote = smote.fit_resample(X, y)

df = X_smote
df['y'] = y_smote

# plt.hist(df['y'])
# plt.title('target variable after over_sampling using SMOTE')
# plt.savefig('figure/y_after_smote.png')
# plt.show()

df.to_csv('data/cleaned_data/bank-full-balanced.csv')