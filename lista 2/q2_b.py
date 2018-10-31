from random import uniform
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import math

pi = math.pi

x = np.zeros((10000,1),dtype=np.float64)
y = np.zeros((10000,1),dtype=np.float64)

for i in range(10000):
    x[i] = uniform(0,4*pi)
    y[i] = (math.cos(2*pi*x[i])/(1 - 16*x[i]*x[i]))*math.sin(pi*x[i])/(pi*x[i])
    
x = np.array(x)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7,), random_state=0)
clf.fit(X_train,y_train) 

predict = clf.predict(x.astype(float))

cont = 0
for i in range(10000):
        