#-------------------------------------------------------------------------
# AUTHOR: Dexxer Medina
# FILENAME: association_rule_mining.py
# SPECIFICATION: Reads in transaction data from a csv file 
#                and creates association rules
# FOR: CS 4200-01 Assignment #5
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

#Use the command: "pip install mlxtend" on your terminal to install the mlxtend library

#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)

#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
#Example:

#Bread Wine Eggs
#1     0    1
#0     1    1
#1     1    1

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)
#-->add your python code below

encoded_vals = []
for index, row in df.iterrows():
    row_list = row.tolist()
    if 'Diaper' in row_list:
        diaper = 1
    else:
        diaper = 0
    
    if 'Eggs' in row_list:
        eggs = 1
    else:
        eggs = 0
        
    if 'Meat' in row_list:
        meat = 1
    else:
        meat = 0
        
    if 'Cheese' in row_list:
        cheese = 1
    else:
        cheese = 0
        
    if 'Pencil' in row_list:
        pencil = 1
    else:
        pencil = 0
        
    if 'Milk' in row_list:
        milk = 1
    else:
        milk = 0
        
    if 'Bagel' in row_list:
        bagel = 1
    else:
        bagel = 0
        
    if 'Wine' in row_list:
        wine = 1
    else:
        wine = 0
        
    if 'Bread' in row_list:
        bread = 1
    else:
        bread = 0

    labels = {'Diaper': diaper, 'Eggs': eggs, 'Meat': meat, 'Cheese': cheese, 'Pencil': pencil, 'Milk': milk, 'Bagel': bagel, 'Wine': wine, 'Bread': bread}

    encoded_vals.append(labels)

#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)


#iterate the rules data frame and print the apriori algorithm results by using the following format:

#Meat, Cheese -> Eggs
#Support: 0.21587301587301588
#Confidence: 0.6666666666666666
#Prior: 0.4380952380952381
#Gain in Confidence: 52.17391304347825
#-->add your python code below
for rule in rules.values:
    ante = list(rule[0])
    cons = list(rule[1])
    supp = rule[4]
    conf = rule[5]
    supportCount = supp * 315
    for x in range(len(ante)):
        print(ante[x], end="")
        if x < len(ante) - 1:
            print(" , ", end="")
    print(" -> ", end="")
            
    for x in cons:
        print(x)
        
    print('Support:', supp)
    print("Confidence:", conf)

#To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
#use the gain formula provided right after.
#prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
#print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
#-->add your python code below
    supportCount = 0
    for i in encoded_vals:
        supportCount += i[cons[0]]
    prior = supportCount/len(encoded_vals)
    print("Prior:", prior)
    
    print("Gain in Confidence: " + str(100*(conf-prior)/prior))
    print()

#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()