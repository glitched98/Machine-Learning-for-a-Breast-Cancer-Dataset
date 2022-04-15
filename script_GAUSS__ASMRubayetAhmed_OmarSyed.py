import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv('C:\\Users\\ahmas\\Downloads\\RYERSON\\WINTER 2022\\CPS 844\\A1\\data.csv')

dataset = dataset.drop(["id"], axis = 1)
dataset = dataset.drop(["Unnamed: 32"], axis = 1)

dataset.info()

M = dataset[dataset.diagnosis == "M"]
B = dataset[dataset.diagnosis == "B"]

plt.title("GAUSS: Malignantt vs Benign Tumor")
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.scatter(M.radius_mean, M.texture_mean, color = "red", alpha = 0.6, marker='+')
plt.scatter(B.radius_mean, B.texture_mean, color = "green", alpha = 0.6, marker='.')
plt.legend(labels=['Malignant','Benign'])

dataset.diagnosis = [1 if i == "M" else 0 for i in dataset.diagnosis]

x = dataset.drop(["diagnosis"], axis = 1)
y = dataset.diagnosis.values

x = (x - np.min(x)) / (np.max(x) - np.min(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

nb = GaussianNB()
nb = nb.fit(x_train, y_train)

print("Naive Bayes Accuracy: ",nb.score(x_test, y_test)*100)