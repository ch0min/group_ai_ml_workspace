import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    cross_val_score,
    GridSearchCV,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

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
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(df.values)

# transformation pipelines

num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ]
)
df_num_tr = num_pipeline.fit_transform(df_num)


num_attribs = list(df_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer(
    [("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs)]
)

df_prepared = full_pipeline.fit_transform(df)

# Training model

# Linear regression
lin_reg = LinearRegression()
lin_reg.fit(df_prepared, df_labels)

some_data = df.iloc[:5]
some_labels = df_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

df_predictions = lin_reg.predict(df_prepared)
lin_mse = mean_squared_error(df_labels, df_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

# Decision tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(df_prepared, df_labels)
df_predictions = tree_reg.predict(df_prepared)
tree_mse = mean_squared_error(df_labels, df_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

scores = cross_val_score(
    tree_reg, df_prepared, df_labels, scoring="neg_mean_squared_error", cv=10
)
tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


display_scores(tree_rmse_scores)

lin_scores = cross_val_score(
    lin_reg, df_prepared, df_labels, scoring="neg_mean_squared_error", cv=10
)

lin_rmse_scoring = np.sqrt(-lin_scores)
display_scores(lin_rmse_scoring)
forest_reg = RandomForestRegressor()
forest_reg.fit(df_prepared, df_labels)
forest_mse = mean_squared_error(df_labels, df_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)
scores = cross_val_score(
    forest_reg, df_prepared, df_labels, scoring="neg_mean_squared_error", cv=10
)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)

# grid search

param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(
    forest_reg,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
)

grid_search.fit(df_prepared, df_labels)

print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importance = grid_search.best_estimator_.feature_importance
print(feature_importance)

extra_attribs = ["rooms_perhhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importance, attributes), reverse=True)

final_model = grid_search.best_estimator_


X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
final = np.sqrt(
    stats.t.interval(
        confidence,
        len(squared_errors) - 1,
        loc=squared_errors.mean(),
        scale=stats.sem(squared_errors),
    )
)
print(final)

# exercises:
# 1. Try a Support Vector Machine regressor (sklearn.svm.SVR) with various hyperparameters, such as kernel="linear" (with various values for the C and gamma hyperparameters). Dont worry about what these hyperparameters mean for now. How does the best SVR predictor perform?
# 2. Try replacing GridSearchCV with RandomizedSearchCV.
# 3. Try adding a transformer in the preparation pipeline to select only the most important attributes
# 4. Try creating a single pipeline that does the full data preparation plus the final prediction
# 5. Automatically explore some preparation options using GridSearchCV
