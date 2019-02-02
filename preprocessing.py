import pandas as pd
import numpy as np
from datetime import timedelta


def load_data():
    data = pd.read_csv("data/BADS_WS1819_known.csv", sep=",")

    col_info = {}
    for column in data.columns:
        col_info[column] = {"type": data[column].dtype}
        col_info[column].update({
            "# of nans": data[column].isnull().sum().sum()})

    #  convert columns
    data["delivery_date"] = pd.to_datetime(
        data["delivery_date"])
    data["item_size"] = data["item_size"].str.upper()
    data["item_size"] = data["item_size"].astype("category")
    data["order_date"] = pd.to_datetime(
        data["order_date"])
    data["user_reg_date"] = pd.to_datetime(
        data["user_reg_date"])
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

    data["delivery_time"] = data[
        "delivery_date"] - data["order_date"]

    # set all user_dob to null if (ordering date - user_dob) > threshold
    data["user_dob"] = data["user_dob"].astype("datetime64")
    age = (data["order_date"] - data["user_dob"]) / np.timedelta64(1, "Y")

    data["user_age"] = age.round()

    data.loc[
            (data["user_age"] < 15) |
            (data["user_age"] > 90),
            ["user_age"]] = np.nan

    data.loc[
            (data["user_age"].isnull()),
            ["user_age"]] = data["user_age"].mean().round()

    # clean up item_size
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

    # group rare user titles
    data.loc[
        data.user_title != "Mrs",
        ["user_title"]] = "Other"
    data["user_title"].astype("category")
    
    # make dummy variables
    tmp = pd.get_dummies(data.user_title)
    data["Mrs"] = tmp["Mrs"]
    data["Other"] = tmp["Other"]

    return data
