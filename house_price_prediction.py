#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

#import dataset
sk_price_dataset = sklearn.datasets.fetch_california_housing()
#print(sk_price_dataset)
house_price_dataset = pd.DataFrame(sk_price_dataset.data, columns = sk_price_dataset.feature_names)
#print(house_price_dataset.head())
house_price_dataset['price'] = sk_price_dataset.target
#print(house_price_dataset.head())

#chech the shape of the dataset
print(house_price_dataset.shape)

#check for missing values
print(house_price_dataset.isnull().sum())

#statistical measure of the dataset
print(house_price_dataset.describe())

#correlation between various data
correlation = house_price_dataset.corr()

#construct a heatmap to understand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True , square=True,fmt = '.1f', annot = True, annot_kws={'size' : 8}, cmap='Blues')
plt.show()

#Splitting the dataset for train and test
X = house_price_dataset.drop(['price'], axis=1)
#print(X.head())
Y = house_price_dataset['price']
#print(Y.head())
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
#print(X.shape, X_train.shape, X_test.shape)

#Model training 
model = XGBRegressor()
model.fit(X_train, Y_train)

#prediction on training data
train_prediction = model.predict(X_train)

#Evaluation

#R squared error
score_1 = metrics.r2_score(Y_train, train_prediction)
#Mean absolute error
score_2 = metrics.mean_absolute_error(Y_train, train_prediction)

print('R sqared error (train): ', score_1)
print('Mean absolute error (train): ', score_2)



#prediction on test data
test_prediction = model.predict(X_test)

#Evaluation
 
#R squared error
score_3 = metrics.r2_score(Y_test, test_prediction)
#Mean absolute error
score_4 = metrics.mean_absolute_error(Y_test, test_prediction)

print('R sqared error (test): ', score_3)
print('Mean absolute error (test): ', score_4)

#Visualize the actual price and predicted price
#plt.scatter(Y_train, train_prediction)
#plt.xlabel("Actual price")
#plt.ylabel("Predicted price")
#plt.title("Actual price vs Predicted price")
#plt.show()


#Making a predictive system
input_data = (8.3252,41.0, 6.984127,1.023810,322.0,2.555556,37.88,-122.23)
#change input data as numpy array
input_data_as_numpy_array  = np.asarray(input_data)
#print(input_data_as_numpy_array)
#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#print(input_data_reshaped)
prediction = model.predict(input_data_reshaped)
print('train price is: 4.526; predictive price is:  ', prediction)




