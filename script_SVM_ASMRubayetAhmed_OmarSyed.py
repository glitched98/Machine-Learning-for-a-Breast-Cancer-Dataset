import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('C:\\Users\\ahmas\\Downloads\\RYERSON\\WINTER 2022\\CPS 844\\A1\\data.csv')

dataset['diagnosis'] = dataset['diagnosis'].map({
    'M': 1,
    'B': 2
})
labels = dataset['diagnosis'].tolist()
dataset['Class'] = labels
dataset = dataset.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1)
dataset.head()

target_names = ['', 'M', 'B']
dataset['attack_type'] = dataset.Class.apply(lambda x: target_names[x])
dataset.head()
dataset.info()
dataset1 = dataset[dataset.Class == 1]
dataset2 = dataset[dataset.Class == 2]

plt.title("Malignant vs Benign Tumor")
plt.xlabel('Radius Mean')
plt.ylabel('Texture Mean')
plt.scatter(dataset1['radius_mean'], dataset1['texture_mean'], color='red', alpha = 0.6, marker='+')
plt.scatter(dataset2['radius_mean'], dataset2['texture_mean'], color='green', alpha = 0.6, marker='.')
plt.legend(labels=['Malignant','Benign'])

X = dataset.drop(['Class', 'attack_type'], axis='columns')
X.head()

y = dataset.Class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(len(X_train))
print(len(X_test))

model = SVC(kernel='linear')

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)

percentage = model.score(X_test, y_test)

res = confusion_matrix(y_test, predictions)
print("Confusion Matrix")
print(res)
print(f"Test Set: {len(X_test)}")
print(f"SVM Accuracy: {percentage*100} %")

