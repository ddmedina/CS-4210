#-------------------------------------------------------------------------
# AUTHOR: Dexxer Lansk Medina
# FILENAME: find_s.py
# SPECIFICATION: Given a training set, use the Find S algorithm to come 
#                up with the Maximally Specific Hypothesis for the data
# FOR: CS 4200-01 Assignment #1
# TIME SPENT: 7 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
import csv

num_attributes = 4
db = []
print("\n The Given Training Data Set \n")

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

print("\n The initial value of hypothesis:\n")
hypothesis = ['0'] * num_attributes #representing the most specific possible hypothesis
print(hypothesis)

#find the first positive training data in db and assign it to the vector hypothesis
##--> add your Python code here
positive_test_cases = []
for test_case in db:
    if test_case[4] == "Yes":
        positive_test_cases.append(test_case)

hypothesis = positive_test_cases[0]
hypothesis.pop()

#find the maximally specific hypothesis according to your training data in db and assign it to the vector hypothesis (special characters allowed: "0" and "?")
##--> add your Python code here
for test_case in positive_test_cases:
    for attribute_index in range(3):
        if test_case[attribute_index] != hypothesis[attribute_index]:
            hypothesis[attribute_index] = '?'

print("\n The Maximally Specific Hypothesis for the given training examples found by Find-S algorithm:\n")
print(hypothesis)
