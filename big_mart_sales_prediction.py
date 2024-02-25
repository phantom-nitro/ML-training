import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

big_mart_data = pd.read_csv('Datasets/big_mart.csv')
print(big_mart_data.head())

#number of data points and number of features
print(big_mart_data.shape)

#getting some information
print(big_mart_data.info())

#check for missing values
print(big_mart_data.isnull().sum())

#handling missing values
print(big_mart_data['Item_Weight'].mean())
#filling the missing values in Item_Weight column with the mean value
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(),inplace=True)
print(big_mart_data.isnull().sum())

#replace the missing values in Outlet_Size column
mode_of_outlet_size = big_mart_data.pivot_table(values = 'Outlet_Size',columns = 'Outlet_Type', aggfunc=(lambda x: x.mode()[0]))

print(mode_of_outlet_size)

missing_values = big_mart_data['Outlet_Size'].isnull()
print(missing_values)

big_mart_data.loc[missing_values, 'Outlet_Size'] = big_mart_data.loc[missing_values, 'Outlet_Type'].apply(lambda x: mode_of_outlet_size)
print(big_mart_data.isnull().sum())

#data analysis
print(big_mart_data.describe())

#numerical features
sns.set()
#Item_Weight distribution
plt.figure(figsize = (6,6))
sns.distplot(big_mart_data['Item_Weight'])
plt.show()
#Item_Visibility distribution
plt.figure(figsize = (6,6))
sns.distplot(big_mart_data['Item_Visibility'])
plt.show()

#Item_Fat_Content distribution
plt.figure(figsize = (6,6))
sns.countplot(x = 'Item_Fat_Content', data = big_mart_data)
plt.show()

print(big_mart_data['Item_Fat_Content'].value_counts())

big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat', 'LF':'Low Fat', 'reg': 'Regular'}}, inplace = True)
print(big_mart_data['Item_Fat_Content'].value_counts())

big_mart_data['Outlet_Size'] = big_mart_data['Outlet_Size'].astype(str)
#Label Encoding
encoder = LabelEncoder()
big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])
big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])
big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])
big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])
big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])
big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])


print(big_mart_data.head())

#Splitting dependent and independent variable
X = big_mart_data.drop(columns = 'Item_Outlet_Sales', axis = 1)
Y = big_mart_data['Item_Outlet_Sales']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state = 3)

#Model training
model = XGBRegressor()
model.fit(X_train, Y_train)

#Evaluation
#Prediction on training data
training_data_predict = model.predict(X_train)
#R square value
train_score = metrics.r2_score(Y_train, training_data_predict)
print('Training r2 score:' , train_score)

#Evaluation
#Prediction on test data
test_data_predict = model.predict(X_test)
#R square value
test_score = metrics.r2_score(Y_test, test_data_predict)
print('Training r2 score:' , test_score)


