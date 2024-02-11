import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#download stopwords
#import nltk
#nltk.download('stopwords')

#print stopwords for english
#print(stopwords.words('english'))

#fake news link: https://www.kaggle.com/datasets/algord/fake-news?resource=download
#1 is real and 0 is fake.

#load dataset
news_dataset = pd.read_csv('Datasets/FakeNewsNet.csv')
#print(news_dataset.shape)
#print(news_dataset.head())
#count missing values
#print(news_dataset.isnull().sum())

#replace null values with empth string
news_dataset = news_dataset.fillna(' ')
#print(news_dataset.isnull().sum())

#combine title and source column
news_dataset['content'] = news_dataset['title'] + ' ' + news_dataset['source_domain']
#print(news_dataset['content'][0])


#stemming
port_stem = PorterStemmer()
def stemming(context):
    stemmed_content = re.sub('[^a-zA-Z]', ' ',context)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content= [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)
#print(news_dataset['content'])

#separating data and label
X = news_dataset['content'].values
Y = news_dataset['real']

#converting text to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
#print(X)


#split train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

#training model
model = LogisticRegression()
model.fit(X_train, Y_train)

#accuracy score

X_train_prediction = model.predict(X_train)
train_accuracy = accuracy_score(X_train_prediction, Y_train)
print('training accuracy: ' , train_accuracy)



X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, Y_test)
print('test accuracy: ', test_accuracy)
