import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# Task 2: Feature Engineering and Descriptive Statistics
# Objective: Enhance the dataset with new features and then use descriptive statistics to explain the distribution of the data.


# import dataframe from cleaned dataset
df = pd.read_pickle("data/interim/cleaned.pkl")
df.info()
df.head()
df.tail()

# fix column Dtypes
df['Date'] = pd.to_datetime(df['Date'], format='%B %d, %Y', errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')


# 1. Feature Engineering:
# - Create a new feature by binning a numerical variable into categories (e.g., low, medium, high). Put ranges (eg age, into three or four groups rather than a continuous distribution)

# age ranges
bins = [0,  20,  40,  60,  80]

df['AgeGroup'] = pd.cut(df['Age'], bins=bins)
age_group_counts = df.groupby('AgeGroup').size()
age_group_counts.plot(kind='bar')
plt.show()

# amount of death by year(from 1922 - 2021)
dategroups = df.groupby(pd.Grouper(key='Date', freq='10Y')).size().reset_index(name='count')
dategroups.set_index('Date', inplace=True)

plt.plot(dategroups.index, dategroups["count"])
plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.show()

# - Implement one-hot encoding for a categorical variable.
df_encoded = pd.get_dummies(df, columns=["Cause of death"])

# 2. Descriptive Statistics:
# - Calculate the mean, median, and standard deviation for numerical features.
numerical_features = df[['Age']]
mean = numerical_features.mean()
median = numerical_features.median()
std_dev = numerical_features.std()

# - For categorical features, count the frequency of each category.
expedition_counts = df['Expedition'].value_counts()
nationality_counts = df['Nationality'].value_counts()
cause_counts = df['Cause of death'].value_counts()
location_counts = df['Location'].value_counts()

# 3. Visualization:
# - Use seaborn to create box plots for numerical features to identify outliers.
for column in ['Age']:
    plt.figure(figsize=(8,  6))
    sns.boxplot(x=df[column])
    plt.title(column)
    plt.show()

# - Visualize the distribution of categorical features using bar plots.
for column in ['Expedition', 'Nationality', 'Cause of death', 'Location']:
    plt.figure(figsize=(8,  6))
    sns.countplot(x=df[column])
    plt.title(column)
    plt.show()