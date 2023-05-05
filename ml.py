from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale  
import numpy
import os
import pandas as pd  
from collections import deque

class MachineLearningAlgo:
    def __init__(self):

        #self.clf = svm.SVC(kernel="linear")
        #self.clf = svm.SVC()
        #self.clf = svm.SVC(gamma=2, C=1)

        #Decision Tree
        #self.clf = tree.DecisionTreeClassifier()
        #self.clf = tree.DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)

        #Gaussian Naive Bayes
        #self.clf = GaussianNB()   

        #Forests of randomized trees
        self.clf = RandomForestClassifier(n_estimators=10)

        #Extra Tree Classifier
        #self.clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)



        #self.clf = MLPClassifier(hidden_layer_sizes=(7), activation="logistic", solver='sgd', beta_1=0.9, beta_2=0.9,
        #                         learning_rate="constant", learning_rate_init=0.1, momentum=0.9)
        # train the model - y values are locationed in last (index 3) column
        X_train = pd.read_csv('result.csv')
        y_train = X_train["type"]
        del X_train["type"]
        X_train.iloc[:] = minmax_scale(X_train.iloc[:])
        self.clf.fit(X_train, y_train.values.ravel())  
        

    def classify(self, data):      
        prediction = self.clf.predict(data)
        print("prediction result ", prediction)
        return prediction
