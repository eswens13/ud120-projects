#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
print "Preprocessing email data . . ."
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Create and train the naive Bayes classifier.
classifier = GaussianNB()
print "\nTraining the naive Bayes classifier . . ."
classifier.fit(features_train, labels_train)
print "\tDone training."

# Make some predictions based on a test data set using the classifier.
t0 = time()
predictions = classifier.predict(features_test)
print "\ntraining time: ", round(time()-t0, 3), "s"

# Calculate the accuracy of the classifier.
t0 = time()
accuracy = accuracy_score(labels_test, predictions, normalize=True)
print "\nprediction time: ", round(time()-t0, 3), "s"

print "\nNB Accuracy: ", accuracy

#########################################################


