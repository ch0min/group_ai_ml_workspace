import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

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

df.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=df["population"] / 100,
    label="population",
    figsize=(10, 7),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
)

corr_matrix = df.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)

# feature engineering
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_rooms"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]


df = strat_train_set.drop("median_house_value", axis=1)
df_labels = strat_train_set["median_house_value"].copy()

# data cleaning

# missing values in total_bedrooms:
# 1. get rid of the corresponding districts
df.dropna(subset=["total_bedrooms"])
# 2 get rid of the whole attribute
df.drop("total_bedrooms", axis=1)
# 3 set the values to some value(zeo, the mean, the median, etc.)
median = df["total_bedrooms"].median()
df["total_bedrooms"].fillna(median, inplace=True)

# using sklearn imputer
imputer = SimpleImputer(strategy="median")

df_num = df.drop("ocean_proximity", axis=1)

imputer.fit(df_num)


imputer.statistics_

df_num.median().values

X = imputer.transform(df_num)

df_tr = pd.DataFrame(X, columns=df_num.columns, index=df_num.index)


# categorical attributes
df_cat = df[["ocean_proximity"]]
df_cat.head()

ordinal_encoder = OrdinalEncoder()
df_cat_encoded = ordinal_encoder.fit_transform(df_cat)
df_cat_encoded[:10]

ordinal_encoder.categories_

# one-hot encoding

cat_encoder = OneHotEncoder()
df_cat_1hot = cat_encoder.fit_transform(df_cat)

df_cat_1hot
df_cat_1hot.toarray()

cat_encoder.categories_

# custom transformers

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, rooms_ix]
        if self.add_bedrooms_per_room:
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(df.values)
