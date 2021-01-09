from _04_preprocessing import housing_prepared, housing_labels
from _04_preprocessing import full_pipeline
from _04_preprocessing import start_test_set

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring='neg_mean_squared_error',
                          return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

best_model = grid_search.best_estimator_


from _04_preprocessing import num_attribs

extra_attribs = ["rooms_per_house", "pop_per_house", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["tf_cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attribs = num_attribs + extra_attribs + cat_one_hot_attribs