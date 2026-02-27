import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

 
data = pd.read_csv(
    r"C:\Data Science\Project Assignments and Project Topic list\Sales Forecasting\stores_sales_forecasting.csv",
    encoding='latin1'
)

 
numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit', 'Postal Code']

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col].fillna(data[col].mean(), inplace=True)
 
categorical_cols = [
    'Ship Mode','Segment','Country','City',
    'State','Region','Category','Sub-Category'
]

for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

 
data.drop([
    'Row ID','Order ID','Order Date','Ship Date',
    'Customer ID','Customer Name','Product ID','Product Name'
], axis=1, inplace=True)
 
data = pd.get_dummies(data)

 
X = data.drop('Profit', axis=1)
y = data['Profit']

 
model = LinearRegression()
model.fit(X, y)

 
pickle.dump(model, open("profit_model.pkl", "wb"))
pickle.dump(X.columns, open("profit_columns.pkl", "wb"))

print("Model trained successfully")