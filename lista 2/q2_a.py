import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras.callbacks as kc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

x = [[0, 0, 0],[0, 0 ,1],[0, 1 ,0],[0, 1, 1],[1, 0, 0],[1, 0, 1],[1, 1, 0],[1, 1, 1]]
y = [0,1,1,0,1,0,0,1]

x = np.array(x)
y = np.array(y)

modelo = Sequential()
modelo.add(Dense(20, input_dim=3, activation='relu'))
modelo.add(Dense(10, init='uniform', activation='relu'))
modelo.add(Dense(1, init='uniform', activation='sigmoid'))

modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

fit = modelo.fit(x, y, epochs=500, batch_size=2, verbose=2)

plt.plot(fit.history['loss'])
plt.title('Analise de desempenho')
plt.ylabel('erro')
plt.xlabel('epocas')
plt.savefig('q2_a.png')
##plt.show()

