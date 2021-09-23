#-------------------------------------------------------------------------
# AUTHOR: Dexxer Lansk Medina
# FILENAME: naive_bayes.py
# SPECIFICATION: use naive bayes model to predict class of training data
#                and output predictions with confidence >= 0.75
# FOR: CS 4210-01 Assignment #2
# TIME SPENT: 30 min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data
#--> add your Python code 
db = [] 

with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append (row)

#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X =
X = []
for training_case in db:
    case = []
    for attr_index in range(1,5):
        # Outlook
        if training_case[attr_index] == 'Sunny':
            case.append(1)
        elif training_case[attr_index] == 'Overcast':
            case.append(2)
        elif training_case[attr_index] == 'Rain':
            case.append(3)
        
        # Temperature
        elif training_case[attr_index] == 'Hot':
            case.append(1)
        elif training_case[attr_index] == 'Mild':
            case.append(2)
        elif training_case[attr_index] == 'Cool':
            case.append(3)
            
        # Humidity
        elif training_case[attr_index] == 'High':
            case.append(1)
        elif training_case[attr_index] == 'Normal':
            case.append(2)
            
        # Wind
        elif training_case[attr_index] == 'Weak':
            case.append(1)
        elif training_case[attr_index] == 'Strong':
            case.append(2)
    
    X.append(case)

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =
Y = []
for training_case in db:
    if training_case[5] == 'Yes':
        Y.append(1)
    elif training_case[5] == 'No':
        Y.append(2)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the data in a csv file
#--> add your Python code here
db_test = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db_test.append (row)

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
#--> add your Python code here
#-->predicted = clf.predict_proba([[3, 1, 2, 1]])[0]
for testcase in db_test:
    case = []
    for attr_index in range(1,5):
        # Outlook
        if testcase[attr_index] == 'Sunny':
            case.append(1)
        elif testcase[attr_index] == 'Overcast':
            case.append(2)
        elif testcase[attr_index] == 'Rain':
            case.append(3)
        
        # Temperature
        elif testcase[attr_index] == 'Hot':
            case.append(1)
        elif testcase[attr_index] == 'Mild':
            case.append(2)
        elif testcase[attr_index] == 'Cool':
            case.append(3)
            
        # Humidity
        elif testcase[attr_index] == 'High':
            case.append(1)
        elif testcase[attr_index] == 'Normal':
            case.append(2)
            
        # Wind
        elif testcase[attr_index] == 'Weak':
            case.append(1)
        elif testcase[attr_index] == 'Strong':
            case.append(2)
    
    predicted = clf.predict_proba([case])[0]
    
    predicted_class = None
    confidence = 0
    if predicted[0] > predicted[1]:
        predicted_class = 'Yes'
        confidence = predicted[0]
    elif predicted[0] < predicted[1]:
        predicted_class = 'No'
        confidence = predicted[1]

    if confidence >= 0.75:
        str_confidence = str(round(confidence, 5))
        print (str(testcase[0]).ljust(15) + str(testcase[1]).ljust(15) + str(testcase[2]).ljust(15) + str(testcase[3]).ljust(15) + str(testcase[4]).ljust(15) + predicted_class.ljust(15) + str_confidence.ljust(15))
