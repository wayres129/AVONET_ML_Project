from math import exp
import random
import csv
random.seed(1)

# Calculate logistic
def logistic(x):
    s = 1 / (1 + exp(-x))
    return s

# Calculate dot product of two lists
def dot(x, y):
    s = 0 # dot product over x and y
    for i in range(len(x)):
        s += (x[i]*y[i])
    return s

# Calculate prediction based on model
def predict(model, point):
    p = logistic(dot(model,point['features'])) # prediction value returned from logistic function
    return p

def load_csv(filename):
    lines = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            lines.append(line)
    return lines