import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("data.csv")
data = data.drop(["Unnamed: 32"], axis = 1)
data = data.drop(["id"], axis = 1)

mal = data[data.diagnosis == "M"]
print(mal.head(5))
beg = data[data.diagnosis == "B"]
print(beg.head(5))

data.diagnosis = [1 if i == "M" else 0 for i in data.diagnosis]

y = data.diagnosis.values
x = data.drop(["diagnosis"], axis = 1)

x = (x - np.min(x)) / (np.max(x) - np.min(x))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

decTree = DecisionTreeClassifier()
decTree.fit(x_train, y_train)
print("Accuracy:", decTree.score(x_test, y_test), "%")

plt.xlabel("Radius")
plt.ylabel("Texture")
plt.scatter(mal.radius_mean, mal.texture_mean, color = "red", label = "Malignant", alpha = 0.3)
plt.scatter(beg.radius_mean, beg.texture_mean, color = "blue", label = "Benign", alpha = 0.3)
plt.legend()
plt.show()
tree.plot_tree(decTree, fontsize = 6)
#print(data.head(15))

