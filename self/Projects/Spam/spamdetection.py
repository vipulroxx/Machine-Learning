# '''
# Solution
# '''
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('SMSSpamCollection.csv',
                   sep='\t', 
                   header=None, 
                   names=['label', 'sms_message'])

df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.shape)
print (df.head())# Output printing out first 5 columns
'''
Here we will look to create a frequency matrix on a smaller document set to make sure we understand how the 
document-term matrix generation happens. We have created a sample document set 'documents'.
'''
# documents = ['Hello, how are you!',
#                 'Win money, win from home.',
#                 'Call me now.',
#                 'Hello, Call hello you tomorrow?']

# count_vector = CountVectorizer()
# print count_vector
# count_vector.fit(documents)
# count_vector.get_feature_names()
# doc_array = count_vector.transform(documents).toarray()
# frequency_matrix = pd.DataFrame(doc_array, columns = count_vector.get_feature_names())
# print frequency_matrix

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=0)

print('total set: {} '.format(df.shape))
print('X training set: {}'.format(X_train.shape))
print('X test set: {}'.format(X_test.shape))
print('Y testing set: {}'.format(y_train.shape))
print('Y test set: {}'.format(y_test.shape))

'''
[Practice Node]

The code for this segment is in 2 parts. Firstly, we are learning a vocabulary dictionary for the training data 
and then transforming the data into a document-term matrix; secondly, for the testing data we are only 
transforming the data into a document-term matrix.

This is similar to the process we followed in Step 2.3

We will provide the transformed data to students in the variables 'training_data' and 'testing_data'.
'''

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)
print ('/////////////////////')
print y_test
print ('/////////////////////')



from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
print naive_bayes.fit(training_data, y_train)



predictions = naive_bayes.predict(testing_data)
print predictions



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions))) #print('Accuracy score: ', format(accuracy_score(y_test, predictions,normalize=False))) fornumber of predictions matched
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))





