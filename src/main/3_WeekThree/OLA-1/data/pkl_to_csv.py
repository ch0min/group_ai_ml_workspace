import pandas as pd

file_path = 'interim/task1_data_processed.pkl'
df = pd.read_pickle(file_path)

csv_file_path = 'raw/task1_data_processed.csv'
df.to_csv(csv_file_path, index=False)
