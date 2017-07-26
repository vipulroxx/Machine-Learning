from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LinearRegression as lreg
housing_data = datasets.load_boston()
lregmodel = lreg()
lregmodel.fit(housing_data.data, housing_data.target)
predictions = lregmodel.predict(housing_data.data)
score = metrics.r2_score(housing_data.target,predictions)
print score
