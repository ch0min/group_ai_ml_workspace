import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Loading dataframes - one for aggregating and one for predicting incidents and ignoring plot warning
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")
df = pd.read_pickle("../OLA-1/data/interim/task2_data_processed.pkl")
df_linear = pd.read_pickle("../OLA-1/data/interim/task2_data_processed.pkl")

# Task 3.1: Selecting the columns I'm going to use and then im aggregating the data
relevant_columns = df[['Age_Group', 'Nationality', 'Cause of death']]

aggregated_data = relevant_columns.groupby(['Age_Group', 'Nationality']).size().reset_index(name='Incident Count')

age_group_incidents = relevant_columns.groupby('Age_Group').size().reset_index(name='Incident Count')
age_group_most_incidents = age_group_incidents.sort_values(by='Incident Count', ascending=False).head(1)

print("\nAge group with the most incidents historically:")
print(age_group_most_incidents)
print("\n")

# Task 3.1: Using linear regression to predict incidents in the upcoming years
df_linear['Year'] = pd.to_datetime(df_linear['Date']).dt.year
yearly_incidents = df_linear.groupby('Year').size().reset_index(name='Incidents')

X = yearly_incidents['Year'].values.reshape(-1, 1)
y = yearly_incidents['Incidents']

model = LinearRegression()
model.fit(X, y)

future_years = np.array(range(int(X.max()) + 1, int(X.max()) + 11)).reshape(-1, 1)
future_predictions = model.predict(future_years)

predictions_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted Incidents': future_predictions})

print(predictions_df)

# Task 3.2:

sns.scatterplot(data=yearly_incidents, x='Year', y='Incidents')
plt.title('Scatter Plot of Year vs Incidents')
plt.xlabel('Year')
plt.ylabel('Number of Incidents')
plt.show()

sns.pairplot(yearly_incidents)
plt.suptitle('Pairwise Relationships in Yearly Incidents Data', verticalalignment='top')
plt.show()
