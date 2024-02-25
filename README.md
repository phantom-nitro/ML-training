# mlp

## svm_classification.py
### model:
  - SVC(kernel='linear', C=1)
### Dataset: 
  - iris data
### Visualization/statistical measure of data: 
  - sns.Facetgrid
### Evaluation: 
  - metrics.classification_report (has precision,recall,f1-score)
  - metrics.confusion_matrix

## diabetes_prediction.py
### model:
  - svm.SVC(kernel="linear")
### Visualization/statistical measure of data: 
  - dataset.describe() (produces count,mean,std,min,25%,50%,75%,max values for all columns)
  - dataset["Outcome"].value_counts() - for each value in Outcome column it counts the number of occurrence
  - dataset.groupby("Outcome").mean() - calculates mean of all column relative to Outcome column values
### data preprocessing:
  - StandardScaler() - makes the data scale-free
### Evaluation: 
  - accuracy_score()

## house_price_prediction.py
### model:
  - XGBRegressor()
### Dataset: 
  - sklearn.datasets.fetch_california_housing()
### data preprocessing:
  - house_price_dataset.isnull().sum() - counts the number of missing values in all columns
### Visualization/statistical measure of data:
  - house_price_dataset.describe()
  - house_price_dataset.corr() - correlation between all columns. if 2 columns increase they are highly correlated.
  - sns.heatmap(correlation, cbar=True , square=True,fmt = '.1f', annot = True, annot_kws={'size' : 8}, cmap='Blues') - visualize the correlation between all columns
### Evaluation: 
  - metrics.r2_score(Y_train, train_prediction) - root squared error
  - metrics.mean_absolute_error(Y_train, train_prediction) - mean absolute error

## fake_news_prediction.py
### model:
  - LogisticRegression()
### data preprocessing:
  - news_dataset.isnull().sum() - counts the number of missing values in all columns
  - news_dataset = news_dataset.fillna(' ') - replace null values with empth string
  - PorterStemmer() - reduce a word to it's root word
  - stopwords.words('english') - remove stopwords words that have no value like 'the' 'a'
  - vectorizer = TfidfVectorizer() - converting text to numerical data (term frequency-inverse document frequency) quantify the importance or relevance of string representations (words, phrases, lemmas, etc) in a document amongst a collection of documents
### Evaluation: 
  - accuracy_score(X_train_prediction, Y_train)

## loan_status_prediction.py
### model:
  - svm.SVC(kernel="linear")
### Visualization/statistical measure of data:
  - loan_dataset["Dependents"].value_counts()
  - sns.countplot(x="Education", hue = "Loan_Status", data = loan_dataset)
### data preprocessing:
  - loan_dataset.isnull().sum() - counts the number of missing values in all columns
  - loan_dataset.dropna() - removes null value rows
  - loan_dataset.replace({"Married":{"No":0,"Yes":1}, "Gender":{"Male":1,"Female":0},"Self_Employed":{"No":0,"Yes":1}, "Property_Area":{"Rural":0,"Semiurban":1,"Urban":2}, "Education":{"Graduate":1, "Not Graduate":0}},inplace=True) - convert categorical columns to numerical values
### Evaluation: 
  - accuracy_score(X_train_prediction, Y_train)

## car_price_prediction.py
### model:
  - LinearRegression()
  - Lasso()
### Visualization/statistical measure of data:
  - car_dataset.info()
  - car_dataset.describe()
  - car_dataset.isnull().sum()
  - car_dataset["Fuel_Type"].value_counts())
### data preprocessing:
  - car_dataset.replace({"Fuel_Type":{"Petrol":0,"Diesel":1,"CNG":2}},inplace=True) - replace categorical data to numerical data
### Evaluation: 
  - linear_test_score=metrics.r2_score(Y_test,linear_test_pred)
  - lasso_test_score=metrics.r2_score(Y_test,lasso_test_pred)

## gold_price_prediction.py
### model:
  - RandomForestRegressor(n_estimators=100)
### Visualization/statistical measure of data:
  - gold_price.describe()
  - sns.heatmap(correlation,cbar=True, square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Reds')
  - sns.distplot(gold_price['GLD'],color='red') - Distribution plot for Gold
  - plt.plot(Y_test,color='red',label='Actual value')
  - plt.plot(test_prediction,color='green',label='Predicted value')
### Evaluation: 
  - test_error_score=metrics.r2_score(Y_test,test_prediction)

## credit_card_fraud_detection.py
### dataset:
  - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
### model:
  - LogisticRegression()
### Visualization/statistical measure of data:
  - creditcard_dataset.describe()
  - creditcard_dataset.info()
### data preprocessing:
  - legit= creditcard_dataset[creditcard_dataset['Class'] == 0]
  - fraud= creditcard_dataset[creditcard_dataset['Class'] == 1]
  - legit_undersample = legit.sample(n=492) - Undersample data
### Evaluation: 
  - train_data_accuracy = metrics.accuracy_score(train_data_prediction,Y_train)
  - test_data_accuracy = metrics.accuracy_score(test_data_prediction,Y_test)

## medical_insurance_prediction.py
### model:
  - LinearRegression()
### Visualization/statistical measure of data:
  - insurance.info()
  - insurance.describe()
  - sns.distplot(insurance['age'])
  - sns.countplot(x='sex',data=insurance)
  - sns.distplot(insurance['bmi'])
  - sns.distplot(insurance['children'])
  - sns.countplot(x='smoker',data=insurance)
### data preprocessing:
  - insurance.replace({'sex':{'male':1,'female':0},'smoker':{'yes':1,'no':0},'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)
### Evaluation: 
  - train_accuracy = metrics.r2_score(Y_train,train_prediction)
  - test_accuracy = metrics.r2_score(Y_test,test_prediction)

## big_mart_sales_prediction.py
### model:
  - XGBRegressor()
### Visualization/statistical measure of data:
  - sns.distplot(big_mart_data['Item_Weight'])
  - sns.distplot(big_mart_data['Item_Visibility'])
  - sns.countplot(x = 'Item_Fat_Content', data = big_mart_data)
### data preprocessing:
  - big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(),inplace=True)
  - big_mart_data.loc[missing_values, 'Outlet_Size'] = big_mart_data.loc[missing_values, 'Outlet_Type'].apply(lambda x: mode_of_outlet_size)
  - big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat', 'LF':'Low Fat', 'reg': 'Regular'}}, inplace = True)
  - encoder = LabelEncoder()
  - big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
### Evaluation: 
  - test_score = metrics.r2_score(Y_test, test_data_predict)
