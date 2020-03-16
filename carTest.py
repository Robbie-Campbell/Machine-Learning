import tensorflow
import keras
import sklearn
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Read the contents of the car.data file.
data = pd.read_csv("car.data")

# The preprocessor is a fast way of converting all String
# variables into a format the machine can understand
le = preprocessing.LabelEncoder()

# Declare all of the data types as variables and as an array
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))


predict = "class"

# Classify the 2 axis into a readable data type
X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(cls)

# Define the learning process and the testing process
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# Define how many closest neighbors are to be calculated
model = KNeighborsClassifier(n_neighbors=12)

# Determine the accuracy of the tests
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "very good"]
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], " Data: ", x_test[x], " Actual: ", y_test[x])