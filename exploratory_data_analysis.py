import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# pandas options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# load data
df = pd.read_csv('data/bank-full.csv', delimiter=';')

# # Exploratory Data Analysis
# # age
# plt.subplots(figsize=(10,5))
# plt.hist(df['age'], bins=30)
# plt.title("Bank Client's Age Histogram")
# plt.savefig("figure/age_histogram.png")
# plt.show()

# # job category
# job_cat = df['job'].value_counts().sort_values()

# plt.subplots(figsize=(10,8))
# plt.barh(y=job_cat.index, width=job_cat.values)
# plt.title('Number of Bank Client by Job')
# plt.savefig("figure/number_of_by_job.png")
# plt.show()

# # marital status
# mar_cat = df['marital'].value_counts().sort_values()

# plt.subplots(figsize=(10,5))
# plt.barh(y=mar_cat.index, width=mar_cat.values)
# plt.savefig("figure/number_of_by_marital_status.png")
# plt.title('Number of Bank Client by Marital Status')


# # housing loan
# housing = df['housing'].value_counts()

# plt.subplots(figsize=(10,5))
# plt.bar(x=housing.index, height=housing.values)
# plt.title('Number of Bank Client by Housing Loan')
# plt.savefig("figure/number_of_by_housing_loan.png")
# plt.show()

# # personal loan
# personal_loan = df['loan'].value_counts()

# plt.subplots(figsize=(10,5))
# plt.bar(x=personal_loan.index, height=personal_loan.values)
# plt.title('Number of Bank Client by Personal Loan')
# plt.savefig("figure/number_of_by_personal_loan.png")
# plt.show()

# # contact
# contact = df['contact'].value_counts()

# plt.subplots(figsize=(10,5))
# plt.bar(x=contact.index, height=contact.values)
# plt.suptitle('Related to The Campaign')
# plt.title('Number of Bank Client by Contact Communication Type')
# plt.savefig("figure/number_of_by_contact_used.png")
# plt.show()

# # previous campaign outcame
# target = df['poutcome'].value_counts()

# plt.subplots(figsize=(10,5))
# plt.bar(x=target.index, height=target.values)
# plt.title('Number of Bank Clients by Previous Campaign Outcome')
# plt.savefig("figure/number_of_by_poutcome.png")
# plt.show()

# # target
# target = df['y'].value_counts()

# plt.subplots(figsize=(10,5))
# plt.bar(x=target.index, height=target.values)
# plt.title('Number of Subscriber of Term Deposit \n (Campaign Target)')
# plt.savefig("figure/number_of_by_y.png")
# plt.show()

# # Avg. Balance
# plt.subplots(figsize=(10,5))
# plt.hist(df['balance'])
# plt.title("Bank Client's Balance Histogram")
# plt.savefig("figure/balance_histogram.png")
# plt.show()

# # Balance Boxplot
# plt.subplots(figsize=(15,5))
# plt.boxplot(df['balance'], vert=0)
# plt.savefig("figure/balance_box_plot_many_outliers.png")
# plt.show()

print(df.describe())

# Boxplot
plt.subplots(figsize=(15,5))
plt.boxplot(df['duration'], vert=0)
plt.savefig("figure/duration_box_plot.png")
plt.show()