import pandas as pd
import numpy as np
from pprint import pprint
import seaborn as sns
from datetime import timedelta  
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


##### START PRE PROCESSING #####

data = pd.read_csv("data/BADS_WS1819_known.csv", sep=",")
to_predict = pd.read_csv("data/BADS_WS1819_unknown.csv", sep=",")

# are Id's unique in both sets ? Yes
# len(set(np.unique(to_predict["order_item_id"]))) == len(np.unique(
    # to_predict["order_item_id"]))
# len(set(np.unique(data["order_item_id"]))) == len(np.unique(
    # data["order_item_id"]))

col_info = {}
for column in data.columns:
    col_info[column] = {"type": data[column].dtype}
    col_info[column].update({
        "# of nans": data[column].isnull().sum().sum()})

# pprint(col_info)

#  convert columns

data["delivery_date"] = pd.to_datetime(
    data["delivery_date"])
data["item_size"] = data["item_size"].str.upper()
data["item_size"] = data["item_size"].astype("category")
data["order_date"] = pd.to_datetime(
    data["order_date"])
data["user_reg_date"] = pd.to_datetime(
    data["user_reg_date"])
data["user_title"].astype("category")
data["user_state"].astype("category")

# the only datapoints, where order_date > delivery_date,
# are the 1994 deliveries
x = data["delivery_date"] < data["order_date"]
x.value_counts()
data.loc[x]["delivery_date"].unique()

# avg delivery time of correct data
data_correct = data[data["delivery_date"] > data["order_date"]]
data_correct = data_correct[data_correct["delivery_date"].notnull()]
data_correct["delivery_time"] = data_correct[
    "delivery_date"] - data_correct["order_date"]
avg_deliv_time = data_correct["delivery_time"].mean()
avg_deliv_time = pd.Timedelta(avg_deliv_time.days, unit="d")

# make arrays with estimated delivery date for every datapoint
# and fill bad and missing values for delivery_date
new_deliv_dates = data.loc[
    (data["delivery_date"] < pd.Timestamp(
        year=2015, month=1, day=1)) |
    (data["delivery_date"].isnull())
    , ["order_date"]] + timedelta(days=avg_deliv_time.days)

data.loc[
        (data["delivery_date"] < pd.Timestamp(
            year=2015, month=1, day=1)) |
        (data["delivery_date"].isnull()),
        ["delivery_date"]] = np.array(new_deliv_dates)

# all good :)
data[data["delivery_date"] > "2016"]
data['delivery_date'].isnull().value_counts()

# maybe do extra area for feature engineering
data["delivery_time"] = data[
    "delivery_date"] - data["order_date"]

# need to clean user_dob

data["order_date"].max()
# set all user_dob to null if (ordering date - user_dob) > threshold
no_null_bday = data["order_date"]
data["user_dob"] = data["user_dob"].astype("datetime64")
age = (data["order_date"] - data["user_dob"]) / np.timedelta64(1, "Y")
age_no_nulls = age.dropna()
# age_no_nulls.groupby(age_no_nulls).count().plot()
# graphical analysis shows many outliers
# make null all where 15 < age < 90
data["user_age"] = age.round()

data.loc[
        (data["user_age"] < 15) |
        (data["user_age"] > 90),
        ["user_age"]] = np.nan
# 11 % nans
data['user_age'].isnull().value_counts(normalize=True) * 100

data.loc[
        (data["user_age"].isnull()),
        ["user_age"]] = data["user_age"].mean().round()
# need to clean item sizes
# rows with item_size == unsized split from rest
unsized = data[data["item_size"].isin(["UNSIZED", "XXL"])]
rest = data[~data["item_size"].isin(["UNSIZED", "XXL"])]
rest['item_size'].apply(len).max()
rest["tmp_len"] = rest["item_size"].apply(len)

rest.loc[
        (rest["tmp_len"] > 3),
        ["item_size"]] = np.nan
rest["item_size"].isnull().value_counts()
rest = rest.dropna()
rest = rest.drop(columns=["tmp_len"])
data = pd.concat([rest, unsized])
data["del_time_int"] = data["delivery_time"].dt.days
data["account_age"] = data.order_date - data.user_reg_date
data[["account_age", "order_date", "user_reg_date"]]

# group rare user titles
data.loc[
    data.user_title != "Mrs",
    ["user_title"]] = "Other"
# make dummy variables
tmp = pd.get_dummies(data.user_title)
data["Mrs"] = tmp["Mrs"]
data["Other"] = tmp["Other"]

one_encode = OneHotEncoder(sparse=False)
one_encode.fit(data.user_title.values)
enc_user_title = one_encode.transform(data.user_title)
#### END PRE PROCESSING #####

##### START EDA #####
# 71
len(data[data["user_title"] == "not reported"]["user_title"])

# no point realy in checking corr, as corr is only checked for int values
cols = pd.Series(data.columns)
no_id_columns = cols[cols.str.contains("_id") == False]
no_id_data = data[no_id_columns]
corr = no_id_data.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

sns.countplot(x="user_title", data=data)

data.groupby("user_title")["order_item_id"].count()
data['user_title'].value_counts(normalize=True) * 100
data['item_size'].value_counts(normalize=True) * 100
data['user_state'].value_counts(normalize=True) * 100
data['item_color'].value_counts(normalize=True) * 100
data['delivery_date'].value_counts(normalize=True) * 100
data.delivery_date.dt.to_period("Y").value_counts(normalize=True) * 100
data['delivery_date'].isnull().value_counts(normalize=True) * 100
data['user_dob'].isnull().value_counts(normalize=True) * 100


data.select_dtypes(include=[np.number])
no_id_data.select_dtypes(include=[np.number]).describe()
sns.boxplot(y="item_price", data=data)
data.loc[data["item_price"] > 89.9, data["item_price"]]
len(data[data["item_price"] > 350]["item_price"])
sns.distplot(data["item_price"])

data[data["delivery_date"] == "1994-12-31"]
