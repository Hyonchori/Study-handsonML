import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from pprint import pprint

csv_dir = "csv"
csv_name = "housing.csv"
csv_path = os.path.join(csv_dir, csv_name)

data = pd.read_csv(csv_path)
print(data.head())
print("")
print(data.info())
print("")
print(data["ocean_proximity"].value_counts())
print("")
print(data.describe())

#data.hist(bins=50, figsize=(16, 12))


from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

data_with_id = data.reset_index()
print(data_with_id.info())
train_set, test_set = split_train_test_by_id(data_with_id, 0.2, "index")

data_with_id["new_id"] = data["longitude"]*1000 + data["latitude"]
print(data_with_id.info())


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

median_income = data["median_income"]
#median_income.hist()
#plt.show()

data["income_cat"] = pd.cut(data["median_income"], bins=[0, 1.5, 3.0, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])
#income_cat = data["income_cat"].hist()
#plt.show()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data["income_cat"]):
    start_train_set = data.loc[train_index]
    start_test_set = data.loc[test_index]

print(start_test_set["income_cat"].value_counts())
print(data["income_cat"].value_counts())

for set_ in (start_train_set, start_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

print(start_train_set)


train_copy = start_train_set.copy()
'''train_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

train_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                s=train_copy["population"]/100, label="population", figsize=(10, 7),
                c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True
                )
plt.legend()
plt.show()'''

corr_matrix = train_copy.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
print("")


from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
'''scatter_matrix(train_copy[attributes], figsize=(12, 8))
plt.show()'''


###############################################################

train_copy["rooms_per_household"] = train_copy["total_rooms"] / train_copy["households"]
train_copy["bedrooms_per_room"] = train_copy["total_bedrooms"] / train_copy["total_rooms"]
train_copy["population_per_household"] = train_copy["population"] / train_copy["households"]

corr_matrix = train_copy.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

###################################################################


train_features = start_train_set.drop("median_house_value", axis=1)
train_labels = start_train_set["median_house_value"].copy()

### option 1 : train_features.dropna(subset=["total_bedrooms"])
### option 2 : train_features.drop("total_bedrooms", axis=1)
### option 3 : median = train_features["total_bedrooms"].median()
############ : train_features["total_bedrooms"].fillna(median, inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
train_features_only_numType = train_features.drop("ocean_proximity", axis=1)
imputer.fit(train_features_only_numType)
print(imputer.statistics_)
print(train_features_only_numType.median().values)

X = imputer.transform(train_features_only_numType)
train_features_tr = pd.DataFrame(X, columns=train_features_only_numType.columns,
                                 index=train_features_only_numType.index)

train_features_only_catType = train_features[["ocean_proximity"]]
print(train_features_only_catType.head(10))


from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
train_features_cat_encoded = ordinal_encoder.fit_transform(train_features_only_catType)
print(train_features_cat_encoded[:10])
print(ordinal_encoder.categories_)


from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
train_features_cat_1hot = cat_encoder.fit_transform(train_features_only_catType)
print(cat_encoder.categories_)
print(train_features_cat_1hot.toarray())


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, pop_ix, house_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_house = X[:, rooms_ix] / X[:, house_ix]
        pop_per_house = X[:, pop_ix] / X[:, house_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_house = X[:, bedrooms_ix] / X[:, house_ix]
            return np.c_[X, rooms_per_house, pop_per_house, bedrooms_per_house]
        else:
            return np.c_[X, rooms_per_house, pop_per_house]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
train_features_axtra = attr_adder.transform(train_features.values)

print(train_features_axtra)


#########################################

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler())
])

train_features_tr = num_pipeline.fit_transform(train_features_only_numType)
print(train_features_tr)


from sklearn.compose import ColumnTransformer

num_attribs = list(train_features_only_numType)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

train_features_prepared = full_pipeline.fit_transform(train_features)
print(train_features_prepared)
print(train_features_prepared.shape)

##############################################



from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(train_features_prepared, train_labels)

some_data = train_features.iloc[:5]
some_labels = train_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print(some_data_prepared.shape)

print(lin_reg.predict(some_data_prepared))
print(list(some_labels))

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

train_predictions = lin_reg.predict(train_features_prepared)
lin_mse = mean_squared_error(train_labels, train_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_mae = mean_absolute_error(train_labels, train_predictions)
print(lin_mse, lin_rmse, lin_mae)


##################################################


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_features_prepared, train_labels)

train_predictions = tree_reg.predict(train_features_prepared)
tree_mse = mean_squared_error(train_labels, train_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_mae = mean_absolute_error(train_labels, train_predictions)
print(tree_mse, tree_rmse, tree_mae)

###################################################


from sklearn.model_selection import cross_val_score

'''scores = cross_val_score(tree_reg, train_features_prepared, train_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print(tree_rmse_scores)

def display_scores(scores):
    print("")
    print("점수 : ", scores)
    print("평균 : ", scores.mean())
    print("표준편차 : ", scores.std())

display_scores(tree_rmse_scores)'''

######################################################

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(train_features_prepared, train_labels)

'''forest_scores = cross_val_score(forest_reg, train_features_prepared, train_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)'''

################################################

from sklearn.model_selection import GridSearchCV

param_grid = [
    {"n_estimators": [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(train_features_prepared, train_labels)
print(grid_search.best_params_)
print("")
print(grid_search.best_estimator_)
print(grid_search.cv_results_)
print("")
feature_importances = grid_search.best_estimator_.feature_importances_
pprint(feature_importances)

extra_attribs = ["rooms_per_house", "pop_per_house", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attribs = num_attribs + extra_attribs + cat_one_hot_attribs
pprint(sorted(zip(feature_importances, attribs), reverse=True))

#####################################################

final_model = grid_search.best_estimator_

test_features = start_test_set.drop("median_house_value", axis=1)
test_labels = start_test_set["median_house_value"].copy()

test_features_prepared = full_pipeline.transform(test_features)

final_predictions = final_model.predict(test_features_prepared)

final_mse = mean_squared_error(test_labels, final_predictions)
final_rmse = np.sqrt(final_mse)
print("")
print(final_rmse)

from scipy import stats

confidence = 0.95
squared_error = (final_predictions - test_labels) **2
print(np.sqrt(stats.t.interval(confidence, len(squared_error)-1,
                               loc=squared_error.mean(),
                               scale=stats.sem(squared_error))))