from _02_making_test_set import start_train_set, start_test_set

housing = start_train_set.drop("median_house_value", axis=1)
housing_labels = start_train_set["median_house_value"].copy()

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

rooms_ix, bedrooms_ix, pop_ix, house_ix = 3, 4, 5, 6

class CombinedAttribsAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_house = X[:, rooms_ix] / X[:, house_ix]
        pop_per_house = X[:, pop_ix] / X[:, house_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_house, pop_per_house, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_house, pop_per_house]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttribsAdder()),
    ("std_scaler", StandardScaler()),
])

housing_num = housing.drop("ocean_proximity", axis=1)

num_attribs = list(housing_num) # names of num_columns
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("tf_num", num_pipeline, num_attribs),
    ("tf_cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)