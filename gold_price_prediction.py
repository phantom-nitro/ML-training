import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

gold_price = pd.read_csv("Datasets/gld_price_data.csv")

#print first 5 rows
print(gold_price.head())
#print last 5 rows
print(gold_price.tail())

#basic info of data
print(gold_price.info())
print(gold_price.isnull().sum())

#getting statistical measures of the data
print(gold_price.describe())

#correlation between various columns in the dataset
correlation = gold_price.drop(columns='Date',axis=1).corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True, square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Reds')
plt.show()
#print correlation values for Gold
print(correlation['GLD'])

#Distribution plot for Gold
sns.distplot(gold_price['GLD'],color='red')
plt.show()

#serate dependent variable and independent vatiable
X=gold_price.drop(columns=['Date','GLD'],axis=1)
Y=gold_price['GLD']

#splitting training and testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=2)

#train model
model=RandomForestRegressor(n_estimators=100)
model.fit(X_train,Y_train)

#model predict
test_prediction=model.predict(X_test)
test_error_score=metrics.r2_score(Y_test,test_prediction)
print("R square error: ",test_error_score)

#compare actual value and predicted value
Y_test = list(Y_test)
#print(Y_test)
plt.plot(Y_test,color='red',label='Actual value')
plt.plot(test_prediction,color='green',label='Predicted value')
plt.xlabel('Number of values')
plt.ylabel('GLD price')
plt.title('Actual vs predicted price')
plt.show()
