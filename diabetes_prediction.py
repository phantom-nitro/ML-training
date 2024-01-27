#import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


#read data and analyze
dataset = pd.read_csv("Datasets/diabetes.csv")
print(dataset.head())

#number of rows and columns in the dataset
print(dataset.shape)

#getting the statistical measures of the data
print(dataset.describe())
print(dataset["Outcome"].value_counts())
print(dataset.groupby("Outcome").mean())

#separating the data
X = dataset.drop(columns = "Outcome", axis = 1)
Y = dataset["Outcome"]
print(X.head())
print(Y.head())

#data Standardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)

X = standardized_data

#train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#training the model
classifier = svm.SVC(kernel="linear")
#training the Support Vector Machine classifier
classifier.fit(X_train, Y_train)

#Model evaluation
#Accuracy score of training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy score of training data: ", training_data_accuracy)

#Accuracy score of test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy score of test data: ", test_data_accuracy)


#Making a predictive system
input_data = (7,196,90,0,0,39.8,0.451,41)
#change input data as numpy array
input_data_as_numpy_array  = np.asarray(input_data)
#print(input_data_as_numpy_array)
#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#print(input_data_reshaped)

#standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)


prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")
