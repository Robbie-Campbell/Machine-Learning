import tensorflow
import keras
import sklearn
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Read the content of the student data
data = pd.read_csv("student-mat.csv", sep=";")

# Select which variables you are measuring
data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]

# Select which variable you want to select
predict = "G3"

# Tidy the data in the file
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)


# Create a variable as a condition for saving a particular
# run of the data
'''
best = 0
# Run the test 30 times
for _ in range(30):

    # Use the sklearn library to split the data into a
    # readable format
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    # Finds a relationship between the entered data
    linear = linear_model.LinearRegression()

    # Tells the variable *linear* what to compare
    linear.fit(x_train, y_train)

    # Measures the accuracy of the tests
    acc = linear.score(x_test, y_test)

    # Saves a value if it's accuracy is the highest out of 30
    if acc > best:
        best = acc
        print(acc)
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
'''
# Opens the highest valued result
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# Displays a prediction of the end result
prediction = linear.predict(x_test)

# Writes the predictions and variables based on the results
for x in range(len(prediction)):
    print(prediction[x], x_test[x], y_test[x])

# Creates 2 variables for use as x and y axis in a visual grid
p = "studytime"
q = "G3"

# Tells the compiler to use the pyplot style
style.use("ggplot")

# Spreads the data according to 2 data types
pyplot.scatter(data[p], data[q])

# Define the x and y axis
pyplot.xlabel(p)
pyplot.ylabel("Final grade")

# Show the result
pyplot.show()
