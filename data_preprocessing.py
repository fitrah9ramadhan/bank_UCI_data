import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')


# pandas set_option
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

df = pd.read_csv("data/bank-full.csv", delimiter=";")

# Numerical Cols

def detect_outliers(numbers, threshold):

	mean = np.mean(numbers)
	std = np.std(numbers)

	outliers = []
	for i in numbers:

		z_score = (i - mean)/std
		if np.abs(z_score) > threshold:
			outliers.append(i)

	return outliers


def remove_outliers(data, threshold):
	
	for col in data.columns:

		if ((data[col].dtype == 'int64') or (data[col].dtype == 'float64')):

			outlier_list_in_col = detect_outliers(numbers=data[col], threshold=threshold)
			data = data[~data[col].isin(outlier_list_in_col)]

	return data


new_df = remove_outliers(data=df, threshold=3)

# Min Max Scaler for high Scaled Data
numerical_col = [col for col in new_df.columns if ((new_df[col].dtype == 'float64') or (new_df[col]).dtype == 'int64')]
to_scale_cols = [col for col in numerical_col if (len(new_df[col].unique()) >= len(new_df[col])*0.01)] # scale if the number of unique value is greater or equal than the length of the rows

scaler = MinMaxScaler()
scaled = scaler.fit(new_df[to_scale_cols])
new_df[to_scale_cols] = scaled.transform(new_df[to_scale_cols])

# Categorical Cols

# Mapping yes or no columns
def yes_no_encoding(data):

	yes_no = set(['yes', 'no'])
	yes_no_cols = []

	categorical_cols = [col for col in data.columns if data[col].dtype == 'object']

	for col in categorical_cols:

		yes_no_founded = set()
		for i in data[col]:

			yes_no_founded.add(i)

		if (yes_no_founded == yes_no):

			yes_no_cols.append(col)

	yes_no_dct = {'no':0, 'yes':1}

	for col in yes_no_cols:
		data[col] = data[col].map(yes_no_dct)

	return data

new_df = yes_no_encoding(data=new_df)

# High Cardinality Columns -> Ordinal Encoding
categorical_cols = [col for col in new_df.columns if new_df[col].dtype == 'object']
high_cardinality_cols = []

for col in categorical_cols:
	if len(new_df[col].unique()) > 5:
		high_cardinality_cols.append(col)

ordinal_encoder = OrdinalEncoder()
ordinal_cols = pd.DataFrame(data=ordinal_encoder.fit_transform(new_df[high_cardinality_cols]),
							index=new_df[high_cardinality_cols].index, 
							columns=high_cardinality_cols)

new_df[high_cardinality_cols] = ordinal_cols
# Low Cardinality Cols -> One Hot Encoding
new_df = pd.get_dummies(new_df)

# print(new_df.info())
# print(new_df.describe())
# print(new_df.shape)

# # Save new_df
new_df.to_csv("data/cleaned_data/bank-full-ready.csv")