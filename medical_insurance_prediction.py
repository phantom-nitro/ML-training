import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
insurance = pd.read_csv('Datasets/insurance.csv')
print(insurance.head())

#number of rows and columns
print(insurance.shape)

#getting some info of data
print(insurance.info())
#check for null values
print(insurance.isnull().sum())

#statistical measures of the dataset
print(insurance.describe())

#distribution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance['age'])
plt.title('Age distribution')
plt.show()

#count plot of Gender value
plt.figure(figsize=(6,6))
sns.countplot(x='sex',data=insurance)
plt.title('Gender count')
plt.show()

#distribution of bmi value
plt.figure(figsize=(6,6))
sns.distplot(insurance['bmi'])
plt.title('BMI distribution')
plt.show()

#distribution of children value
plt.figure(figsize=(6,6))
sns.distplot(insurance['children'])
plt.title('Children distribution')
plt.show()

#count plot of smoker value
plt.figure(figsize=(6,6))
sns.countplot(x='smoker',data=insurance)
plt.title('Smoker count')
plt.show()

#encoding the categorical features
insurance.replace({'sex':{'male':1,'female':0},'smoker':{'yes':1,'no':0},'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)

print(insurance.head())

#splitting dependent and independent variable
X = insurance.drop(columns='charges',axis=1)
Y = insurance['charges']

#split train and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=2)

#train the model
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#train predict
train_prediction = regressor.predict(X_train)
train_accuracy = metrics.r2_score(Y_train,train_prediction)
print('train r2 score: ' , train_accuracy)

#test predict
test_prediction = regressor.predict(X_test)
test_accuracy = metrics.r2_score(Y_test,test_prediction)
print('test r2 score: ' , test_accuracy)


