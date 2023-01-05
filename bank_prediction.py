import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# pandas options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# load data
df = pd.read_csv('data/bank-full.csv', delimiter=';')

# Exploratory Data Analysis
# age
plt.subplots(figsize=(10,5))
plt.hist(df['age'], bins=30)
plt.title("Bank Client's Age Histogram")
# plt.show()
# plt.savefig("figure/age_histogram.png")

# job category
job_cat = df['job'].value_counts().sort_values()

plt.subplots(figsize=(10,8))
plt.barh(y=job_cat.index, width=job_cat.values)
plt.xticks(rotation=45)
plt.title('Number of Bank Client by Job')
# plt.show()
# plt.savefig("figure/number_of_by_job.png")

# marital category
mar_cat = df['marital'].value_counts().sort_values()

plt.subplots(figsize=(10,5))
plt.barh(y=mar_cat.index, width=mar_cat.values)
plt.xticks(rotation=45)
plt.title('Number of Bank Client by Marital Status')
plt.savefig("figure/number_of_by_marital_status.png")