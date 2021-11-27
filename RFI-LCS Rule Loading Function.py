


import numpy as np
import pandas as pd
import ast
from skExSTraCS import ExSTraCS
from skExSTraCS import Classifier
from skExSTraCS import OfflineEnvironment

#data parameter: pandas dataframe of original dataset
#rule_csv: file path of csv containing information for all rules (example included in repository)
#pickle_file: file path of txt used to store rules and reboot LCS 

def RFILCS_Rule_Loading (data, rule_csv, pickle_file, classLabel, number_of_iterations):
    rule_count = rule_csv.shape[0]
    instance_count = data.shape[0]
    attribute_list = []
    for col in data.columns:
        attribute_list.append(col)
        
    rule_accuracy_dict = {}

    for rule in range (0, rule_count):
        match_set = []
        correct_set = []
        for instance in range (0, instance_count):
            match = True
            attribute_index_string = rule_csv.iloc[rule]['Attribute Index']
            attribute_index_list = ast.literal_eval(attribute_index_string)
            condition_string = rule_csv.iloc[rule]['Condition']
            condition_list = ast.literal_eval(condition_string)
        
            for i in range(0, len(attribute_index_list)):
                if data.iloc[instance][attribute_list[attribute_index_list[i]]] not in condition_list[i]:
                    match = False
            
            if match == True:
                match_set.append(instance)
        
            if match == True and data.iloc[instance]['Class'] == rule_csv.iloc[rule]['Class']:
                correct_set.append(instance)
    
        if len(match_set) > 0:
            rule_accuracy_dict[rule] = len(correct_set) / len(match_set)
        elif len(match_set) == 0:
            rule_accuracy_dict[rule] = 0
    
    newPopSet = []

    for rule in range (0, rule_count):
        dummymodel = ExSTraCS()
        newClassifier  = Classifier(dummymodel)
        attribute_index_string = rule_csv.iloc[rule]['Attribute Index']
        attribute_index_list = ast.literal_eval(attribute_index_string)
        newClassifier.specifiedAttList = attribute_index_list
    
        condition_string = rule_csv.iloc[rule]['Condition']
        condition_list = ast.literal_eval(condition_string)
        newClassifier.condition = condition_list
    
        newClassifier.phenotype = rule_csv.iloc[rule]['Class']
        newClassifier.fitness = rule_accuracy_dict[rule]
        newClassifier.accuracy = rule_accuracy_dict[rule]
        newClassifier.numerosity = rule_csv.iloc[rule]['Numerosity']
    
        newClassifier.aveMatchSetSize = 1
        newClassifier.timeStampGA = 0
        newClassifier.initTimeStamp = 0
    
        newPopSet.append(newClassifier)

    
    dataFeatures = data.drop(classLabel,axis = 1).values
    dataPhenotypes = data[classLabel].values
    env = OfflineEnvironment(dataFeatures, dataPhenotypes, dummymodel)    

    dummymodel = ExSTraCS()
    dummymodel.env = OfflineEnvironment(dataFeatures, dataPhenotypes, dummymodel) 
    dummymodel.hasTrained = True
    dummymodel.iterationCount = dummymodel.learning_iterations
    dummymodel.finalMetrics = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, env, newPopSet, newPopSet]
    dummymodel.pickle_model(pickle_file)



    model2 = ExSTraCS(learning_iterations = number_of_iterations,nu=10,N=2000,reboot_filename=pickle_file)
    print("Score with inputted rules")
    print(model2.score(dataFeatures,dataPhenotypes))
    model2.fit(dataFeatures,dataPhenotypes)
    print("Score with LCS after initialization with rules and learning iterations")
    print(model2.score(dataFeatures,dataPhenotypes))




