#-------------------------------------------------------------------------
# AUTHOR: Dexxer Lansk Medina
# FILENAME: decision_tree.py
# SPECIFICATION: Creates a decision tree based on training data
# FOR: CS 4200-01 Assignment #1
# TIME SPENT: 8 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
#--> add your Python code here
# X = 
for test_case in db:
    case = []
    for attr_index in range(4):
        if test_case[attr_index] == 'Young':
            case.append(1)
        elif test_case[attr_index] == 'Prepresbyopic':
            case.append(2)
        elif test_case[attr_index] == 'Presbyopic':
            case.append(3)
        elif test_case[attr_index] == 'Myope':
            case.append(1)
        elif test_case[attr_index] == 'Hypermetrope':
            case.append(2)
        elif test_case[attr_index] == 'No':
            case.append(1)
        elif test_case[attr_index] == 'Yes':
            case.append(2)
        elif test_case[attr_index] == 'Reduced':
            case.append(1)
        elif test_case[attr_index] == 'Normal':
            case.append(2)
    
    X.append(case)
    

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> addd your Python code here
# Y =
for test_case in db:
    if test_case[4] == 'Yes':
        Y.append(1)
    elif test_case[4] == 'No':
        Y.append(2)

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()
