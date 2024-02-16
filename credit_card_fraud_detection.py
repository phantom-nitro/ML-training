import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

#import dataset 
creditcard_dataset = pd.read_csv('Datasets/creditcard.csv')
print(creditcard_dataset.head())
print(creditcard_dataset.describe())
print(creditcard_dataset.info())

#Statistical measure based on the class
print(creditcard_dataset['Class'].value_counts())
print(creditcard_dataset.groupby('Class').mean())

#Under sample
legit= creditcard_dataset[creditcard_dataset['Class'] == 0]
fraud= creditcard_dataset[creditcard_dataset['Class'] == 1]
print(legit.shape)
print(fraud.shape)
legit_undersample = legit.sample(n=492) #random sample
print(legit_undersample.shape)

#combine legit_undersample and fraud data
combined_transactions = pd.concat([legit_undersample,fraud],axis=0)
print(combined_transactions.head())
print(combined_transactions.tail())

#split dependent and independent variable
X = combined_transactions.drop(columns = 'Class',axis = 1)
Y = combined_transactions['Class']

#split train and test data
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)

#model training
model = LogisticRegression()
model.fit(X_train,Y_train)

#predict the training data
train_data_prediction = model.predict(X_train)
train_data_accuracy = metrics.accuracy_score(train_data_prediction,Y_train)
print('Train data accuracy score: ' , train_data_accuracy)

#predict the test data
test_data_prediction = model.predict(X_test)
test_data_accuracy = metrics.accuracy_score(test_data_prediction,Y_test)
print('Test data accuracy score: ' , test_data_accuracy)


