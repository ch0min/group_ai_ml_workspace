import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ************************************************************** #
#                                                                #
#             TASK 3: DATA WRANGLING AND ANALYSIS                #
#                                                                #
# ************************************************************** #


df = pd.read_pickle("../data/interim/task2_data_processed.pkl")


# --------------------------------------------------------------
# 1. Data Selection and Wrangling
# --------------------------------------------------------------

"""
    Hypothetical Question: 
    Did more people die before the 2000's on Mount Everest climbs,
    potentially due to advancements in equipment and techniques?
"""


# Extracting the year
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df["Year"] = df["Date"].dt.year

# Categorizing data into periods
df["Period of Death"] = np.where(df["Year"] < 2000, "Before 2000", "2000 and After")

# Count the frequency of deaths in each period
death_count_by_period = df["Period of Death"].value_counts()
print(death_count_by_period)

plt.figure(figsize=(8, 5))
sns.countplot(x="Period of Death", data=df)
plt.title("Number of Deaths Before and After 2000")
plt.xlabel("Period of Death")
plt.ylabel("Number of Deaths")
plt.savefig("./figures/number_of_deaths_before_2000_and_after.png")
plt.show()


# Creating dataframe for before and after 2000
deaths_before_2000 = df[df["Year"] < 2000]
deaths_after_2000 = df[df["Year"] >= 2000]

# Counting the Causes of Deaths for each period
cod_before_2000 = deaths_before_2000["Cause of death"].value_counts()
cod_after_2000 = deaths_after_2000["Cause of death"].value_counts()

plt.figure(figsize=(12, 22))
sns.barplot(
    x=cod_after_2000.values, y=cod_after_2000.index, color="r", label="2000 and After"
)
sns.barplot(
    x=-cod_before_2000.values, y=cod_before_2000.index, color="b", label="Before 2000"
)

plt.title("Comparison of Causes of Death Before and After 2000")
plt.xlabel("Number of Deaths")
plt.ylabel("Cause of Death")
plt.legend()
plt.savefig("./figures/comparison_of_cod_before_and_after_2000.png")
plt.show()


# Group by year and death frequency
deaths_per_year = df.groupby("Year").size()

plt.figure(figsize=(12, 6))
deaths_per_year.plot(kind="line")
plt.title("Fatalities Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Deaths")
plt.grid(True)
plt.savefig("./figures/deaths_per_year.png")
plt.show()

# Calculate mean Age for each "Cause of death" category:
mean_age_by_cod = df.groupby("Cause of death")["Age"].mean()
print(mean_age_by_cod)


# --------------------------------------------------------------
# 2. Data Analysis
# --------------------------------------------------------------

# Visualizing scatter plot for "Age" and "Year" of death
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Year", y="Age", data=df)
plt.title("Relationship between Age and Year of death")
plt.xlabel("Year")
plt.ylabel("Age")
plt.grid(True)
plt.savefig("./figures/relation_btw_age_and_year.png")
plt.show()

# Pairplot for comparison of Age, Year and Period
selected_columns = ["Age", "Year", "Period of Death"]
pairplot_df = df[selected_columns]

sns.pairplot(pairplot_df, hue="Period of Death")
plt.savefig("./figures/pairplot_age_year_period.png")
plt.show()
