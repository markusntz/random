# perceptron

import numpy as np

class Perceptron(object):

    def __init__(self, input_size, learning_rate=0.02, iterations=100):

        self.weight = np.zeros(input_size)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.input_size = input_size

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = 0
        for i in range(self.input_size):
            z += self.weight[i]*x[i]
        res = self.activation(z)

        return res 
    
    def learn(self, X, y):
        for _ in range(self.iterations):
            for i in range(y.shape[0]):
                x_i = X[i]
                y_i = self.predict(x_i)
                error = y - y_i
                print(error)

                self.weight += self.weight + self.learning_rate * error * y_i
    

X_train = np.array([
    [1,0,0],
    [1,0,1],
    [1,1,0], 
    [1,1,1]
])

y_train = np.array([0, 0, 0, 1])

model = Perceptron(input_size=3)
model.learn(X_train, y_train)

print(model.weight)
