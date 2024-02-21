import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ************************************************************** #
#                                                                #
#      TASK 2: FEATURE ENGINEERING & DESCRIPTIVE STATISTICS      #
#                                                                #
# ************************************************************** #

# Load dataset
df = pd.read_pickle("../OLA-1/data/interim/task1_data_processed.pkl")

df.info()

# --------------------------------------------------------------
# 1. Feature Engineering
# --------------------------------------------------------------


# Defining age bins and labels
def bin_age(
    dataset,
    column_name="Age",
    bins=[0, 20, 40, 60, np.inf],
    labels=["0-20", "21-40", "41-60", "61+"],
):
    """
    Bins the age column of the dataset into specified categories.

    Args:
        dataset: The dataset containing the data
        column_name: The name of the col to be binned.
        bins: The edges of the bins as a list.
        labels: The labels for the bins.

    Returns: A dataset with an additional column for the age_groups.

    """
    dataset["Age_Group"] = pd.cut(
        dataset[column_name], bins=bins, labels=labels, include_lowest=True
    )
    return dataset


df = bin_age(df)

print(df["Age_Group"].value_counts())


# Implementing one-hot encoding for categorical variables

# df["Nationality"].value_counts()
# ohe_nationality = pd.get_dummies(df, columns=["Nationality"])
# ohe_nationality.head(20)

# df["Expedition"].value_counts()
# ohe_expedition = pd.get_dummies(df, columns=["Age_Group"])
# ohe_expedition.head(20)

df["Age_Group"].value_counts()
ohe_age = pd.get_dummies(df, columns=["Age_Group"])
ohe_age.head(20)


# --------------------------------------------------------------
# 2. Descriptive Statistics
# --------------------------------------------------------------

# Calculate the mean, median, and standard deviation for numerical features.
age_descriptive_stats = df["Age"].describe()

# For categorical features, count the frequency of each category.
freq_name = df["Name"].value_counts()
freq_date = df["Date"].value_counts()
freq_age_group = df["Age_Group"].value_counts()
freq_nationality = df["Nationality"].value_counts()
freq_cod = df["Cause of death"].value_counts()
freq_loc = df["Location"].value_counts()


# --------------------------------------------------------------
# 3. Visualization
# --------------------------------------------------------------

# Use seaborn to create box plots for numerical features to identify outliers.
"""
    Identifying outliers of "Age" after filling missing values of "Age",
    will give a skewed representation of outliers, since there was approx. 150 missing
    values from "Age" in the beginning, and we came to the conclusion to fill the
    missing data with the median.
    
    This boxplot will therefore show an imprecise representation of outliers,
    since there's a significant amount that are 38 of age.
    
"""
plt.figure(figsize=(8, 10))
sns.boxplot(y=df["Age"])
plt.title("Boxplot for Age (imprecise)")
plt.xlabel("Age")
plt.savefig("./figures/age_boxplot_imprecise.png")
plt.show()


# Visualize the distribution of categorical features using bar plots.
"""
    We already visualized categorical features using bar plots in
    Task 1: [4. Data Visualization]
    
    Visualizations can be found in ./figures directory.
"""


# Creating pickle file
df.to_pickle("../OLA-1/data/interim/task2_data_processed.pkl")

