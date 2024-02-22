import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import pickle


df = pd.read_pickle("../OLA-1/data/interim/task1_data_processed.pkl")

############################
########## TASK 3 ##########
############################

### 1. Data selection and wrangling ###



# Select a subset of columns relevant to a hypothetical question of 
# interest (e.g., predicting a target variable).

df_subset = df[['Location', 'Cause of death', 'Age']]

# Aggregate data
location_cause_counts = df_subset.groupby(['Location', 'Cause of death']).size().reset_index(name='Count')

# Sort the data (better visualization)
location_cause_counts_sorted = location_cause_counts.sort_values(by='Count', ascending=False)

# Display the top 10 to see the most common combinations
print(location_cause_counts_sorted.head(10))


# Calculate mean 'Age' for each 'Cause of death'
mean_age_by_cause = df_subset.groupby('Cause of death')['Age'].mean().reset_index(name='Mean Age')

# Sort the data by 'Mean Age' for better visualization
mean_age_by_cause_sorted = mean_age_by_cause.sort_values(by='Mean Age', ascending=False)

print(mean_age_by_cause_sorted)


### 2. Data Analysis ###

# Extract the year from the 'Date' column and create a new 'Year' column
df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year

# Handle NaN values in 'Age'
df['Age'] = df['Age'].fillna(df['Age'].median())

#'Age' vs 'Year' - describes how the age of the climbers have varied over the years
sns.scatterplot(x='Year', y='Age', data=df)
plt.xlabel('Year of Expedition')
plt.ylabel('Age of Climber')
plt.title('Scatter Plot of Climber Age over Years')
plt.show()

# create dataframe with numerical only
df_numerical = df[['Age', 'Year']].dropna()

# pairplot with seaborn
sns.pairplot(df_numerical)
plt.savefig('pairplot.png')
plt.show()


### 3.
