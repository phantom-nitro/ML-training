import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

#load dataset
car_dataset = pd.read_csv("Datasets/car data.csv")
print(car_dataset.head())

#check the number of rows and columns
print(car_dataset.shape)
print(car_dataset.info())
print(car_dataset.describe())
print(car_dataset.isnull().sum())

#check number of categorical data
print(car_dataset["Fuel_Type"].value_counts())
print(car_dataset["Seller_Type"].value_counts())
print(car_dataset["Transmission"].value_counts())

#replace categorical data
car_dataset.replace({"Fuel_Type":{"Petrol":0,"Diesel":1,"CNG":2}},inplace=True)
car_dataset.replace({"Seller_Type":{"Dealer":0,"Individual":1}},inplace=True)
car_dataset.replace({"Transmission":{"Manual":0,"Automatic":1}},inplace=True)

#split independent and dependent variable
X=car_dataset.drop(columns=["Car_Name","Selling_Price"],axis=1)
Y=car_dataset["Selling_Price"]

#split data to train and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=2)
print(X.shape,Y_train.shape,Y_test.shape)

#trainning model
#Linear Regression
linear_reg=LinearRegression()
linear_reg.fit(X_train,Y_train)

linear_train_pred=linear_reg.predict(X_train)
linear_train_score=metrics.r2_score(Y_train,linear_train_pred)
print("r2 score linear train prediction: ",linear_train_score)
linear_test_pred=linear_reg.predict(X_test)
linear_test_score=metrics.r2_score(Y_test,linear_test_pred)
print("r2 score linear test prediction: ",linear_test_score)

#Visualization
plt.scatter(Y_test,linear_test_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Actual Price vs Predicted Price")
plt.show()

#Lasso Regression
lasso_reg=Lasso()
lasso_reg.fit(X_train,Y_train)

lasso_train_pred=lasso_reg.predict(X_train)
lasso_train_score=metrics.r2_score(Y_train,lasso_train_pred)
print("r2 score lasso train prediction: ",lasso_train_score)
lasso_test_pred=lasso_reg.predict(X_test)
lasso_test_score=metrics.r2_score(Y_test,lasso_test_pred)
print("r2 score lasso test prediction: ",lasso_test_score)

#Visualization
plt.scatter(Y_test,lasso_test_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Lasso Actual Price vs Predicted Price")
plt.show()
