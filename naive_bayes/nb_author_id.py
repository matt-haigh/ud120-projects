#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

print "Calling preprocess()"

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print "Preprocess is done."


#########################################################
### your code goes here ###

print len(features_train)
print len(features_test)
print len(labels_train)
print len(labels_test)

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

print "Fitting"
pre_fit_time = time()
classifier.fit(features_train, labels_train)
print "training time:", round(time()-pre_fit_time, 3), "s"

print "Predicting"
pre_predict_time = time()
prediction = classifier.predict(features_test)
print "training time:", round(time()-pre_predict_time, 3), "s"

print "accuracy:"
print accuracy_score(labels_test, prediction)

#########################################################


