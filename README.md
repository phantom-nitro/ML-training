# mlp
Evaluation metrics used

svm_classification.py
model:
  SVC(kernel='linear', C=1)
Dataset: 
  iris data
Visualization/statistical measure of data: 
  sns.Facetgrid
Evaluation: 
  metrics.classification_report (has precision,recall,f1-score)
  metrics.confusion_matrix

diabetes_prediction.py
model:
  svm.SVC(kernel="linear")
Visualization/statistical measure of data: 
  dataset.describe() (produces count,mean,std,min,25%,50%,75%,max values for all columns)
  dataset["Outcome"].value_counts() - for each value in Outcome column it counts the number of occurrence
  dataset.groupby("Outcome").mean() - calculates mean of all column relative to Outcome column values
data preprocessing:
  StandardScaler() - makes the data scale-free
Evaluation: 
  accuracy_score()

house_price_prediction.py
model:
  XGBRegressor()
Dataset: 
  sklearn.datasets.fetch_california_housing()
data preprocessing:
  house_price_dataset.isnull().sum() - counts the number of missing values in all columns
Visualization/statistical measure of data:
  house_price_dataset.describe()
  house_price_dataset.corr() - correlation between all columns. if 2 columns increase they are highly correlated.
  sns.heatmap(correlation, cbar=True , square=True,fmt = '.1f', annot = True, annot_kws={'size' : 8}, cmap='Blues') - visualize the correlation between all columns
Evaluation: 
  metrics.r2_score(Y_train, train_prediction) - root squared error
  metrics.mean_absolute_error(Y_train, train_prediction) - mean absolute error

fake_news_prediction.py
model:
  LogisticRegression()
data preprocessing:
  news_dataset.isnull().sum() - counts the number of missing values in all columns
  news_dataset = news_dataset.fillna(' ') - replace null values with empth string
  PorterStemmer() - reduce a word to it's root word
  stopwords.words('english') - remove stopwords words that have no value like 'the' 'a'
  vectorizer = TfidfVectorizer() - converting text to numerical data (term frequency-inverse document frequency) quantify the importance or relevance of string representations (words, phrases, lemmas, etc) in a document amongst a collection of documents
Evaluation: 
  accuracy_score(X_train_prediction, Y_train)
