import pandas as pd
import copy, math
import numpy as np
import matplotlib.pyplot as plt


# plt.style.use('./deeplearning.mplstyle')

np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

df = pd.read_csv("train.csv")   # replace with your filename
df = df.dropna()  # REMOVE missing values
print(df.head())




# X_train = all columns except the target
X_train = df.drop(columns=["close","date","symbols"]).values   # example: drop the output column
# y_train = only the target/output column
y_train = df["close"].values


#print(X_train.shape)
#print(y_train.shape)

#print(X_train[:5])  
#print(y_train[:5])  

# Normalize the data
from sklearn.preprocessing import StandardScaler    

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

#print(X_train[:5])  
#print(X_train_scaled[:5])  

#normalinze them youreself when you have time

# Model training using Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("Model trained.")
print(model.predict(X_train_scaled[:5]))
print(y_train[:5])  # compare with actual value