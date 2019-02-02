from sklearn.model_selection import train_test_split
from preprocessing import load_data
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


data = load_data()

data["del_time_int"] = data["delivery_time"].dt.days
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

# CV model
data_dmatrix = xgb.DMatrix(data=X, label=y)
model = xgb.XGBClassifier()
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
