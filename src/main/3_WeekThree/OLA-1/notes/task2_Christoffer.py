import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset
df = pd.read_pickle("../data/interim/task1_data_processed.pkl")

# Binning the Age into categories and defining bins and labels for the age categories
bins = [0, 30, 50, 100]
labels = ['Low', 'Mid', 'High']
df['Age_category'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Applying one-hot encoding to the "Nationality" variable
df_encoded = pd.get_dummies(df, columns=['Nationality'])

print(df_encoded.head())

# Identify numerical and categorical features
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
categorical_features = df.select_dtypes(include=['object', 'category']).columns

# Calculate mean, median, and standard deviation for the numerical features
numerical_summary = df[numerical_features].agg(['mean', 'median', 'std'])

# Count the frequency of each category for the categorical features
categorical_summary = {feature: df[feature].value_counts() for feature in categorical_features}

# Display the numerical summary
print("Numerical Summary:")
print(numerical_summary)

# Display the categorical summary
print("\nCategorical Summary:")
for feature, summary in categorical_summary.items():
    print(f"\n{feature} Frequency Counts:")
    print(summary)

# Create box plots for numerical features
plt.figure(figsize=(10, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(len(numerical_features), 1, i)
    sns.boxplot(x=df[feature])
    plt.title(feature)
plt.tight_layout()
plt.show()

# Visualize the distribution of categorical features using bar plots
plt.figure(figsize=(10, 8))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(len(categorical_features), 1, i)
    sns.countplot(y=df[feature], order=df[feature].value_counts().index)
    plt.title(feature)
    plt.tight_layout()
plt.show()
