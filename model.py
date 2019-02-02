# import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from preprocessing import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import xgboost as xgb


data = load_data()

data["del_time_int"] = data["delivery_time"].dt.days
# linear model wont really work, as we dont have a lot numerical variables
# split data
numerics = [
    'uint8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_df = data.select_dtypes(include=numerics)
numeric_df = numeric_df.drop(columns=["item_id", "brand_id", "user_id"])
X = numeric_df.loc[:, numeric_df.columns != "return"]
y = numeric_df.loc[:, numeric_df.columns == "return"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Logistic Regression
reg = LogisticRegression().fit(X_train, y_train)
reg.score(X_test, y_test)

# Support Vector Machine
# kinda bad predictions
# svclassifier = SVC(kernel='linear')
# svclassifier.fit(X_train, y_train)

# xg_reg = xgb.XGBRegressor(
#     objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.1,
#     max_depth=5, alpha=10, n_estimators=10
# )

data_dmatrix = xgb.DMatrix(data=X, label=y)

params = {
    "objective": "binary:logistic",
    'colsample_bytree': 0.3,
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha': 10,
    'booster': "dart"
}

cv_results = xgb.cv(
    dtrain=data_dmatrix, params=params, nfold=10, num_boost_round=100,
    early_stopping_rounds=10, metrics="error", as_pandas=True, seed=123)

# configure RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Random Forest
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=100,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1)
rf_random.fit(X_train, y_train)

# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
# metrics.f1_score(y_test, y_pred)
