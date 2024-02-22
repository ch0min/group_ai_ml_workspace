import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Task 3: Data Wrangling and Analysis
# Objective: Perform data wrangling to prepare data for analysis and conduct simple
# analysis to extract stories about the data - what can we say about this data?.

df = pd.read_pickle("../data/interim/task2_data_processed.pkl")
df.info()
df.head()
df.tail()

# Tasks:
# 1. Data Selection and Wrangling:
# - Select a subset of columns relevant to a hypothetical question of interest (e.g., predicting a target variable).
relevant_columns = ['Age', 'Nationality', 'Cause of death', 'Location']
df_relevant = df[relevant_columns]

df["Year"] = df["Date"].dt.year

# - Use .groupby() to aggregate data and calculate mean values for each category of a selected categorical variable.
agg_df = df_relevant.groupby('Nationality')['Age'].count().reset_index(name='Number of deaths')

# 2. Data Analysis:
# - Use seaborn to create scatter plots to visualize relationships between pairs of numerical variables.(X an Y axis are used for the variables)
# Create a scatter plot of Age vs. Year
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Year'], y=df['Age'], data=df_relevant)
plt.title('Age vs. Year')
plt.xlabel('Year')
plt.ylabel('Age')
plt.show()

# - Create a pairplot to visualize the pairwise relationships in the dataset.
# See https://seaborn.pydata.org/generated/seaborn.pairplot.html
plt.figure(figsize=(20, 20))
sns.pairplot(df[['Age', "Year"]])
plt.show()

# 3. Insights:
# - Based on the visualizations and descriptive statistics, write down 3 insights about the dataset:

# - the range of the age of people dying on mount everest has increased. 
# - the max age of people dying on mount everest has increased drastically
# - the fact that the mean of age is overrepresented because of data cleaning skews the results 