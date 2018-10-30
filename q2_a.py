from sklearn.neural_network import MLPClassifier

x = [[0, 0, 0],[0, 0 ,1],[0, 1 ,0],[0, 1, 1],[1, 0, 0],[1, 0, 1],[1, 1, 0],[1, 1, 1]]
y = [0,1,1,0,1,0,0,1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7,), random_state=0)
clf.fit(x, y) 

predict = clf.predict(x)