#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
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
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

from sklearn import svm
from sklearn.metrics import accuracy_score

C_param = 10000;

# Cut down the training data by 99%.
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

# Create SVM classifier and train it.
print "Training SVM classifier with C =", C_param, ". . ."
classifier = svm.SVC(C=10000, kernel="rbf")

# Train the classifier.
t0 = time()
classifier.fit(features_train, labels_train)
print "\tTraining time: ", round(time() - t0, 3), "s"

# Make some predictions
print "Making predictions . . ."
t0 = time()
pred = classifier.predict(features_test)
print "\tPrediction time: ", round(time() - t0, 3), "s"

# Calculate the accuracy of the SVM.
print "Calculating the accuracy of the SVM . . ."
score = accuracy_score(labels_test, pred, normalize=True)
print "\tAccuracy ( C =", C_param, "): ", score

print "Predictions:"
print "\t#10 =", pred[10]
print "\t#26 =", pred[26]
print "\t#50 =", pred[50]

# Count how many of the emails are classified to be authored by Chris (1)
chris_count = 0
for email in pred:
    if email == 1:
        chris_count += 1
print "Number of Chris emails: ", chris_count

