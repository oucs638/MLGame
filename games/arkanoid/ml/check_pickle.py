import pickle
from os import path
import numpy as np
filename = path.join(path.dirname(__file__), 'save',
                     'clf_KNN_BallAndDirection.pickle')
with open(filename, 'rb') as file:
    clf = pickle.load(file)
print(clf)
# for data in clf:
# print(data)
