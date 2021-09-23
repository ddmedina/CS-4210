#-------------------------------------------------------------------------
# AUTHOR: Dexxer Lansk Medina
# FILENAME: decision_tree.py
# SPECIFICATION: trains a decision tree model on 3 separate data sets and 
#                calculates the accuracy of each model
# FOR: CS 4210-01 Assignment #2
# TIME SPENT: 35 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
dataTest = 'contact_lens_test.csv'

for ds in dataSets:

    dbTraining = []
    dbTest = []
    X = []
    Y = []
    NumError = 0
    worst_accuracy = 0
    
    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    # X =
    for test_case in dbTraining:
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
    #--> add your Python code here
    # Y =
    for test_case in dbTraining:
        if test_case[4] == 'Yes':
            Y.append(1)
        elif test_case[4] == 'No':
            Y.append(2)
    
    #loop your training and test tasks 10 times here
    for i in range (10):
        #print("Pass ", i)

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        #--> add your Python code here
        # dbTest =
        with open(dataTest, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append (row)

       
        for data in dbTest:
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
           
            data_attr = []
            for attr_index in range(4):
                if data[attr_index] == 'Young':
                    data_attr.append(1)
                elif data[attr_index] == 'Prepresbyopic':
                    data_attr.append(2)
                elif data[attr_index] == 'Presbyopic':
                    data_attr.append(3)
                elif data[attr_index] == 'Myope':
                    data_attr.append(1)
                elif data[attr_index] == 'Hypermetrope':
                    data_attr.append(2)
                elif data[attr_index] == 'No':
                    data_attr.append(1)
                elif data[attr_index] == 'Yes':
                    data_attr.append(2)
                elif data[attr_index] == 'Reduced':
                    data_attr.append(1)
                elif data[attr_index] == 'Normal':
                    data_attr.append(2)
                   
            class_predicted = clf.predict([data_attr])[0]
                    
            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            true_class = 0
            if data[4] == 'Yes':
                true_class = 1
            elif data[4] == 'No':
                true_class = 2
            
            if class_predicted != true_class:
                NumError += 1
        
        error_rate = NumError / len(dbTest)
        NumError = 0

        #find the lowest accuracy of this model during the 10 runs (training and test set)
        #--> add your Python code here
        if error_rate > worst_accuracy:
            worst_accuracy = error_rate

    #print the lowest accuracy of this model during the 10 runs (training and test set) and save it.
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print("Final accuracy when training on", ds, ":", 1 - worst_accuracy)
