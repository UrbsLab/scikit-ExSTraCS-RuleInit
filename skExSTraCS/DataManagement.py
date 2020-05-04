import numpy as np
import math

class DataManagement:
    def __init__(self,dataFeatures,dataPhenotypes,model):
        self.savedRawTrainingData = [dataFeatures,dataPhenotypes]
        self.numAttributes = dataFeatures.shape[1]  # The number of attributes in the input file.
        self.attributeInfoType = [
                                     0] * self.numAttributes  # stores false (d) or true (c) depending on its type, which points to parallel reference in one of the below 2 arrays
        self.attributeInfoContinuous = [[0, 0]] * self.numAttributes  # stores continuous ranges and NaN otherwise
        self.attributeInfoDiscrete = [0] * self.numAttributes  # stores arrays of discrete values or NaN otherwise.
        for i in range(0, self.numAttributes):
            self.attributeInfoDiscrete[i] = AttributeInfoDiscreteElement()

        # About Phenotypes
        self.discretePhenotype = True  # Is the Class/Phenotype Discrete? (False = Continuous)
        self.phenotypeList = []  # Stores all possible discrete phenotype states/classes or maximum and minimum values for a continuous phenotype
        self.phenotypeRange = None  # Stores the difference between the maximum and minimum values for a continuous phenotype
        self.isDefault = True  # Is discrete attribute limit an int or string
        try:
            int(model.discreteAttributeLimit)
        except:
            self.isDefault = False

        #Initialize some variables
        self.continuousCount = 0
        self.classPredictionWeights = {}
        self.averageStateCount = 0

        # About Dataset
        self.numTrainInstances = dataFeatures.shape[0]  # The number of instances in the training data
        self.discriminateClasses(dataPhenotypes)

        self.discriminateAttributes(dataFeatures, model)
        self.characterizeAttributes(dataFeatures, model)

        #Rule Specificity Limit
        if model.ruleSpecificityLimit == None:
            i = 1
            uniqueCombinations = math.pow(self.averageStateCount,i)
            while uniqueCombinations < self.numTrainInstances:
                i += 1
                uniqueCombinations = math.pow(self.averageStateCount,i)
            model.ruleSpecificityLimit = min(i,self.numAttributes)

        self.trainFormatted = self.formatData(dataFeatures, dataPhenotypes, model)  # The only np array

    def discriminateClasses(self,phenotypes):
        currentPhenotypeIndex = 0
        classCount = {}
        while (currentPhenotypeIndex < self.numTrainInstances):
            target = phenotypes[currentPhenotypeIndex]
            if target in self.phenotypeList:
                classCount[target]+=1
                self.classPredictionWeights[target] += 1
            else:
                self.phenotypeList.append(target)
                classCount[target] = 1
                self.classPredictionWeights[target] = 1
            currentPhenotypeIndex+=1

        total = 0
        for eachClass in list(classCount.keys()):
            total += classCount[eachClass]
        for eachClass in list(classCount.keys()):
            self.classPredictionWeights[eachClass] = 1 - (self.classPredictionWeights[eachClass]/total)

    def discriminateAttributes(self,features,model):
        for att in range(self.numAttributes):
            attIsDiscrete = True
            if self.isDefault:
                currentInstanceIndex = 0
                stateDict = {}
                while attIsDiscrete and len(list(stateDict.keys())) <= model.discreteAttributeLimit and currentInstanceIndex < self.numTrainInstances:
                    target = features[currentInstanceIndex,att]
                    if target in list(stateDict.keys()):
                        stateDict[target] += 1
                    elif np.isnan(target):
                        pass
                    else:
                        stateDict[target] = 1
                    currentInstanceIndex+=1

                if len(list(stateDict.keys())) > model.discreteAttributeLimit:
                    attIsDiscrete = False
            elif model.discreteAttributeLimit == "c":
                if att in model.specifiedAttributes:
                    attIsDiscrete = False
                else:
                    attIsDiscrete = True
            elif model.discreteAttributeLimit == "d":
                if att in model.specifiedAttributes:
                    attIsDiscrete = True
                else:
                    attIsDiscrete = False

            if attIsDiscrete:
                self.attributeInfoType[att] = False
            else:
                self.attributeInfoType[att] = True
                self.continuousCount += 1

    def characterizeAttributes(self,features,model):
        for currentFeatureIndexInAttributeInfo in range(self.numAttributes):
            for currentInstanceIndex in range(self.numTrainInstances):
                target = features[currentInstanceIndex,currentFeatureIndexInAttributeInfo]
                if not self.attributeInfoType[currentFeatureIndexInAttributeInfo]:#if attribute is discrete
                    if target in self.attributeInfoDiscrete[currentFeatureIndexInAttributeInfo].distinctValues or np.isnan(target):
                        pass
                    else:
                        self.attributeInfoDiscrete[currentFeatureIndexInAttributeInfo].distinctValues.append(target)
                        self.averageStateCount += 1
                else: #if attribute is continuous
                    if np.isnan(target):
                        pass
                    elif float(target) > self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][1]:
                        self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][1] = float(target)
                    elif float(target) < self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][0]:
                        self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][0] = float(target)
                    else:
                        pass
            if self.attributeInfoType[currentFeatureIndexInAttributeInfo]: #if attribute is continuous
                self.averageStateCount += 2
        self.averageStateCount = self.averageStateCount/self.numAttributes

    def formatData(self,features,phenotypes,model):
        formatted = np.insert(features,self.numAttributes,phenotypes,1) #Combines features and phenotypes into one array
        np.random.shuffle(formatted)
        shuffledFeatures = formatted[:,:-1].tolist()
        shuffledLabels = formatted[:,self.numAttributes].tolist()
        for i in range(len(shuffledFeatures)):
            for j in range(len(shuffledFeatures[i])):
                if np.isnan(shuffledFeatures[i][j]):
                    shuffledFeatures[i][j] = None
            if np.isnan(shuffledLabels[i]):
                shuffledLabels[i] = None
        return [shuffledFeatures,shuffledLabels]


class AttributeInfoDiscreteElement():
    def __init__(self):
        self.distinctValues = []