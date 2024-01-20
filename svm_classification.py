#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris_data = pd.read_csv("Datasets/iris.csv")

#count of each species
#print(iris_data["Species"].value_counts())

#replacing categorical data
iris_data["Species"].replace(['setosa', 'versicolor', 'virginica'],[0,1,2],inplace=True)
#print(iris_data.head())

#splitting target
X = iris_data[["Petal.Length","Petal.Width"]]
Y = iris_data["Species"]
#visualization of the data
#sns.FacetGrid(iris_data, hue="Species", height=8).map(plt.scatter, "Petal.Length","Petal.Width").add_legend()
#plt.show()

#fit SVM model to data
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1)
model.fit(X,Y)

print("model.score: ")
print(model.score(X,Y))

#make prediction
expected = Y
predicted = model.predict(X)

from sklearn import metrics
#summarize the fit of the model
print("Classification report: ")
print(metrics.classification_report(expected, predicted))
print("Confusion matrix: ")
print(metrics.confusion_matrix(expected, predicted))
