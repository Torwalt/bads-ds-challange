import pandas as pd
import numpy as np
import seaborn as sns
import preprocessing
import matplotlib.pyplot as plt

data = preprocessing.load_data()
data["user_dob"].groupby(data["user_dob"].dt.month).count()
# data["user_dob"].groupby(data["user_dob"].dt.year).count().plot()

# no point realy in checking corr, as corr is only checked for int values
# cols = pd.Series(data.columns)
# no_id_columns = cols[cols.str.contains("_id") == False]
# no_id_data = data[no_id_columns]
# corr = no_id_data.corr()
# sns.heatmap(corr,
            # xticklabels=corr.columns.values,
            # yticklabels=corr.columns.values)
# 
# sns.countplot(x="user_title", data=data)

data.groupby("user_title")["order_item_id"].count()
data['user_title'].value_counts(normalize=True) * 100
data['item_size'].value_counts(normalize=True) * 100
data['item_size'].value_counts().plot()
data['item_size'].apply(len).max()
plt.show()
data['user_state'].value_counts(normalize=True) * 100
data['item_color'].value_counts(normalize=True) * 100
data['delivery_date'].value_counts(normalize=True) * 100
data.delivery_date.dt.to_period("Y").value_counts(normalize=True) * 100
data['delivery_date'].isnull().value_counts(normalize=True) * 100


data.select_dtypes(include=[np.number])
no_id_data.select_dtypes(include=[np.number]).describe()
sns.boxplot(y="item_price", data=data)
data.loc[data["item_price"] > 89.9, data["item_price"]]
len(data[data["item_price"] > 350]["item_price"])
sns.distplot(data["item_price"])

data[data["delivery_date"] == "1994-12-31"]
