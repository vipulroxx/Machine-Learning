

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import pickle
from pandas import DataFrame
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
df = DataFrame(enron_data)
c =0
print list(df.index)
#for names in df.index:
 #   print df[names]['total_payments']
