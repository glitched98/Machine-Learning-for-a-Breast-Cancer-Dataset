import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("data.csv")
#print(data.isnull().sum())

data = data.drop(["Unnamed: 32", "id"], axis =1)
data["diagnosis"] = data["diagnosis"].map({'B':0,'M':1}).astype(int)

x=data[['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean','radius_se','perimeter_se', 'area_se','compactness_se', 'concave points_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','texture_worst','area_worst']]
y=data[['diagnosis']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

model = KNeighborsClassifier(n_neighbors=8)
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy_score(predict,y_test)

acc = model.score(x_train,y_train)
print(data.head(15))
print('The accuracy of the classifier is', acc)
#print(accuracy_score)
print(data.size)
