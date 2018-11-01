from random import uniform
from sklearn.model_selection import train_test_split
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold

def modelo():
    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))
    model.add(Dense(5, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model
pi = math.pi
n = 500
x = np.zeros((n,1),dtype=np.float64)
y = np.zeros((n,1),dtype=np.float64)

for i in range(n):
    x[i] = uniform(0,4*pi)
    y[i] = (math.cos(2*pi*x[i])/(1 - 16*x[i]*x[i]))*math.sin(pi*x[i])/(pi*x[i])
    
x = np.array(x)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)


estimator = KerasRegressor(build_fn=modelo, nb_epoch=500, batch_size=50, verbose=0)
estimator.fit(x,y)
kfold = KFold(n_splits=10, random_state=0)
results = cross_val_score(estimator, x, y, cv=3)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#estimator.fit(X, y)
prediction = estimator.predict(X_test)
accuracy_score(y_test, prediction)

#plt.plot(fit.history['loss'])
#plt.title('Analise de desempenho')
#plt.ylabel('erro')
#plt.xlabel('epocas')
#plt.savefig('q2_b1.png')
#plt.close()
#
#y_estimado = modelo.predict(X_test)
#
#plt.plot(X_test[0:800], y_estimado[0:800], 'g+')
#plt.plot(X_test[0:800], y_test[0:800], 'bx')
#plt.title('Resultado da função')
#plt.savefig('q2_b2.png')
