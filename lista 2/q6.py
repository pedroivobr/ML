import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

def func(n):
    return np.sin(n + np.sin(n)**2)

plt.subplot(221)
x = np.linspace(0, 30, 1000)
y = func(x)
plt.title('Sinal contínuo')
plt.plot(x, y)

plt.subplot(222)
x = np.arange(0, 30).reshape(-1, 1)
y = func(x)
plt.title('Sinal discreto')
plt.stem(x, y)
plt.show()

n=100 
num_pontos = 30
num_amostras = 1000
X_train,y_train = [],[]

for i in range(num_amostras):
    aux = np.arange(n-i-1-num_pontos, n-i-1) # índices temporais
    X_train.append(func(aux))
    y_train.append(func(n-i-1))
X_train = np.array(X_train)
y_train = np.array(y_train)

model = Sequential()
model.add(Dense(60, input_dim=num_pontos, activation='linear'))
model.add(Dense(30, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

parada = EarlyStopping(monitor='loss',patience=10) #numero de epocas sem melhoras
hist = model.fit(X_train, y_train, epochs=30, batch_size=20, verbose=0, callbacks=[parada])

plt.plot(hist.history['loss'])
plt.xlabel('Epocas')
plt.ylabel('Erro')
plt.show()  


aux=np.arange(n-num_pontos,n)
predict = model.predict(np.array([func(aux)]))[0][0]
real = func(n)

print('previsão e real: %.2f e %.2f'%(predict,real))