import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

# load the data
dataset = pd.read_csv('C:\\Users\\ahmas\\Downloads\\RYERSON\\WINTER 2022\\CPS 844\\A1\\data.csv')

dataset = dataset.drop(["id"], axis = 1)
dataset = dataset.drop(["Unnamed: 32"], axis = 1)

dataset.info()

# seperate the class from the other attributes
classData = dataset['diagnosis']
attributeData = dataset.drop(['diagnosis'], axis=1)

# Standardize the data
attributeData = normalize(attributeData)

# Split the data
dataTrain, dataTest, classTrain, classTest = train_test_split(attributeData, classData, test_size = 0.3, random_state = 42)

# KNN classification
clf = MLPClassifier()
clf.fit(dataTrain, classTrain)
predC = clf.predict(dataTest)

# Print accuracy
print("Accuracy:", accuracy_score(classTest, predC)*100, "%")