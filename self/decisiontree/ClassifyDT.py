from sklearn.tree import DecisionTreeClassifier
def classify(features_train, labels_train):

    #### your code goes here -- should return a trained decision tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    return clf
