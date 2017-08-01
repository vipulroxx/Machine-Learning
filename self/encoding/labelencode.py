import pandas
# We import the preprocessing module to manipulate the data
from sklearn import preprocessing
# creating sample data
sample_data = {'name': ['Ray', 'Adam', 'Jason', 'Varun', 'Xiao'],
'health':['fit', 'slim', 'obese', 'fit', 'slim']}
# storing sample data in the form of a dataframe
data = pandas.DataFrame(sample_data, columns = ['name', 'health']) 
label_encoder = preprocessing.LabelEncoder()
#label_encoder.fit(data['health'])
#print label_encoder.transform(data['health'])
#You can combine the fit and transform statements above by using
print label_encoder.fit_transform(data['health'])
