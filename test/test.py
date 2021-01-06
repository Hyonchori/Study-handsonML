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