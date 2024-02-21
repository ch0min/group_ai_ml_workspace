import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

csv_path = "../OLA-1/data/raw/me_climbing_deaths.csv"
data = pd.read_csv(csv_path)

############################
########## TASK 2 ##########
############################

### 1. Binning "age" into young/middle-aged/senior ###

# Define the bins with labels
bins = [0, 30, 50, 100]
labels = ['Young', 'Middle-aged', 'Senior']

# Create new 'Age Group' column
data['Age Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

# create new filtered Data to exclude rows where 'Age' is NaN so
# we can show new age groups
data_non_nan = data.dropna(subset=['Age'])


print(data_non_nan[['Age', 'Age Group']].head())

# Apply one-hot encoding on causes of death
data_encoded = pd.get_dummies(data, columns=['Cause of death'])

print(data_encoded.head())




### 2. Descriptive Statistics ###

# Calculate mean
mean = data.mean(numeric_only=True)

# Calculate median
median = data.median(numeric_only=True)

# Calculate standard deviation
std_dev = data.std(numeric_only=True)

print("Mean Values:\n", mean)
print("\nMedian Values:\n", median)
print("\nStandard Deviation Values:\n", std_dev)


# count frequency for categorial feature (example: cause of death)
cause_of_death_counts = data['Cause of death'].value_counts()

# Display the frequency counts
print("Frequency of Each Category in 'Cause of death':\n", cause_of_death_counts)




### 3. Visualization ###

# Create a box plot for the 'Age' column
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Age'])
plt.title('Box Plot of Age') 
plt.xlabel('Age')
plt.show() 


# Visualize the distribution of causes of death using bar plots.
# Count the frequency of each category in 'Cause of death' and select the top 10
top_causes = data['Cause of death'].value_counts().head(10)

# Create a bar plot for the top 10 causes of death
plt.figure(figsize=(12, 8))
sns.barplot(x=top_causes.values, y=top_causes.index, palette='viridis')
plt.title('Top 10 Causes of Death')
plt.xlabel('Frequency')
plt.ylabel('Cause of Death')
plt.show()
