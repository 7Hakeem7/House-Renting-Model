import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import warnings

warnings.filterwarnings("ignore")

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(r'C:/Users/HP/OneDrive/Desktop/House Renting Model/data/Housing.csv')

# Data Preprocessing
df["mainroad"] = df["mainroad"].replace(["yes", "no"], [1, 0])
df["guestroom"] = df["guestroom"].replace(["yes", "no"], [1, 0])
df["basement"] = df["basement"].replace(["yes", "no"], [1, 0])
df["hotwaterheating"] = df["hotwaterheating"].replace(["yes", "no"], [1, 0])
df["airconditioning"] = df["airconditioning"].replace(["yes", "no"], [1, 0])
df["furnishingstatus"] = df["furnishingstatus"].replace(["furnished", "semi-furnished", "unfurnished"], [2, 1, 0])

# Split the dataset for price prediction into train and test
X1 = df.drop(["price"], axis=1)
y1 = df[["price"]]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)

# Train Linear Regression model
linearregression1 = LinearRegression()
linearregression1.fit(X1_train, y1_train)

# Save the trained model to a file
filename = 'trainedModel.pkl'
pickle.dump(linearregression1, open(filename, 'wb'))

# (Optional) Evaluate the model
pred_train1 = linearregression1.predict(X1_train)
pred_test1 = linearregression1.predict(X1_test)
print("R2 Squared for X1:")
lrscore_train1 = linearregression1.score(X1_train, y1_train)
lrscore_test1 = linearregression1.score(X1_test, y1_test)
print(lrscore_train1)
print(lrscore_test1)

data1 = y1_test.copy()
data1["pred1"] = pred_test1
data1["residual1"] = data1["price"] - data1["pred1"]
print(data1.head())
