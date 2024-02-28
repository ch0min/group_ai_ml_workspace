import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# --------------------------------------------------------------
# 1. Find and Download a Dataset
# --------------------------------------------------------------
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# fetch_housing_data()


# --------------------------------------------------------------
# 2. Data Exploration
# --------------------------------------------------------------
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join("../../", housing_path, "housing.csv")
    return pd.read_csv(csv_path)


df = load_housing_data()
df.head()
df.info()
df["ocean_proximity"].value_counts()
df.describe()

df.hist(bins=50, figsize=(20, 15))


# --------------------------------------------------------------
# 3. Generating Test Set
# --------------------------------------------------------------
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(df, 0.2)

len(train_set)
len(test_set)


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xFFFFFFFF < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# use row index as unique identifier
df_with_id = df.reset_index()
train_set, test_set = split_train_test_by_id(df_with_id, 0.2, "index")


# use row id based on longitude and latitude as unique identifier
df_with_id["id"] = df["longitude"] * 1000 + df["latitude"]
trains_set, test_set = split_train_test_by_id(df_with_id, 0.2, "id")

# use sklearn to split df
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

bins = [0, 1.5, 3, 4.5, 6, np.inf]  # Define your bins here
labels = [1, 2, 3, 4, 5]  # Labels for the bins

# stratified sampling
df["income_cat"] = pd.cut(df["median_income"], bins=bins, labels=labels)
df["income_cat"].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# --------------------------------------------------------------
# 3. Data Visualization
# --------------------------------------------------------------

df = strat_train_set.copy()
