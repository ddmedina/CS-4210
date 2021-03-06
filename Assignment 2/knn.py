#-------------------------------------------------------------------------
# AUTHOR: Dexxer Medina
# FILENAME: knn.py
# SPECIFICATION: Reads in data points and uses KNN classifier to find the 
#                leave-one-out cross-validation error rate for 1NN classifier
# FOR: CS 4210-01 Assignment #2
# TIME SPENT: 20 min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# Only used to suppress warnings from imported SK Learn module
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
trueInstanceClass = None
numErrors = 0.0

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    #--> add your Python code here
    # X =
    X = []
    for j, point in enumerate(db):
        X.append([float(point[0]), float(point[1])])
    X.pop(i)

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    #--> add your Python code here
    # Y =
    Y = []
    for k, point in enumerate(db):
        if point[2] == '-':
            Y.append(1.0)
        elif point[2] == '+':
            Y.append(2.0)
    trueInstanceClass = Y.pop(i)

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    #testSample =
    testSample = [instance[0], instance[1]]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != trueInstanceClass:
        numErrors += 1.0

#print the error rate
#--> add your Python code here
print("Error Rate: ", numErrors / len(db))
