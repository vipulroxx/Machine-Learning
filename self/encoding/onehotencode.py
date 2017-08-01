import pandas
from sklearn import preprocessing
# creating sample data
sample_data = {'name': ['Ray', 'Adam', 'Jason', 'Varun', 'Xiao'],
'health':['fit', 'slim', 'obese', 'fit', 'slim']}
# storing sample data in the form of a dataframe
data = pandas.DataFrame(sample_data, columns = ['name', 'health'])
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(data['health'])

#Using pandas
print pandas.get_dummies(data.health)

#Using sklearn
ohe = preprocessing.OneHotEncoder() # creating OneHotEncoder object
label_encoded_data = label_encoder.fit_transform(data['health'])
print ohe.fit_transform(label_encoded_data.reshape(-1,1))
