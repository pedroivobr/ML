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
import keras.callbacks as callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def baseline_model():
    modelo = Sequential()
    modelo.add(Dense(10, input_dim=1, activation='relu'))
    modelo.add(Dense(5, init='uniform', activation='relu'))
    modelo.add(Dense(1, init='uniform', activation='sigmoid'))
    modelo.compile(loss='mean_squared_error', optimizer='adam')
    return modelo


pi = math.pi
n = 500
x = np.linspace(0, 4*pi, n).reshape(-1, 1)
y = np.zeros((n,1),dtype=np.float64)


standardscaler_X = StandardScaler()
standardscaler_Y = StandardScaler()

x_train = standardscaler_X.fit_transform(x)
y_train = standardscaler_Y.fit_transform(y)

for i in range(n):
    y[i] = (math.cos(2*pi*x[i])/(1 - 16*x[i]*x[i]))*math.sin(pi*x[i])/(pi*x[i])
    
# evaluate model with standardized dataset
#seed=7
#np.random.seed(seed)
#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
#pipeline = Pipeline(estimators)
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(pipeline, x, y, cv=kfold)
#print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))







#from sklearn.preprocessing import LabelEncoder
#labelencoder_X = LabelEncoder()
#labelencoder_Y = LabelEncoder()
#x = labelencoder_X.fit_transform(x)
#y = labelencoder_Y.fit_transform(y)
#
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)
#
#
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=500, batch_size=50, verbose=0)
hfit = estimator.fit(x,y)
kfold = KFold(n_splits=10, random_state=0)
results = cross_val_score(estimator, x, y, cv=3)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

plt.plot(x,y)
plt.xlabel('Época')
plt.ylabel('Custo')
plt.title('Histórico do custo')
plt.show()
#prediction = estimator.predict(X_test)
#accuracy_score(y_test, prediction)

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
