import pandas as pd

file_path = '../OLA-1/data/interim/task1_data_processed.pkl'
df = pd.read_pickle(file_path)

csv_file_path = '../OLA-1/data/raw/task1_data_processed.csv'
df.to_csv(csv_file_path, index=False)
