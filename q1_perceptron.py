import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

x_data = np.array([[0, 0, 0],[1,0,0],[1,0,1],[1,1,0],[0,0,1],[0,1,1],[0,1,0],[1,1,1]])
y_data = np.array([0,0,0,0,1,1,1,1])

n_iter = 3;

n_iter_no_change=5
per = Perceptron(n_iter=n_iter)
per.fit(x_data, y_data)

pred = per.predict(x_data)

print(pred) #valor calculadora para saída