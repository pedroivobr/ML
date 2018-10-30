import numpy as np
from sklearn import svm

x_data = [[0, 0, 0],[1,0,0],[1,0,1],[1,1,0],[0,0,1],[0,1,1],[0,1,0],[1,1,1]]
y_data = [0,0,0,0,1,1,1,1]

help(svm.SVC)

clf = svm.SVC(kernel='linear')
clf.fit( x_data, y_data)

pred = clf.predict(x_data)

print(pred) #valor calculadora para saída