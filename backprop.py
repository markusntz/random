# backprop by hand
import numpy as np

X_train = np.array(([1,0,1], [1,1,0], [1,1,1], [1,0,0]))
y_test = X_train

y_train = np.array(([1], [1], [0], [1]))
y_test = y_train 


class MultiLayerPerceptron(object):


    def __init__(self):
        self.input_size = 3
        self.hidden_size = 3
        self.output_size = 1

        self.weight_jk = np.random.randn(self.input_size, self.hidden_size)
        self.weight_ij = np.random.randn(self.hidden_size, self.output_size)


    def feed_forward(self, X):
        self.a_j = np.dot(X, self.weight_jk)
        self.y_j = self.sigmoid(self.a_j)
        self.y_j[2] = 1.0 # add bias
        self.a_i = np.dot(X, self.weight_ij)

        y_i = self.sigmoid(self.a_i)

        return y_i


    def sigmoid(self, s):
        return 1/(1+np.exp(-s)) 


    def sigmoid_first(self, s):
        s = self.sigmoid(s)
        return s*(1-s)


    def backprop(self, X, y, y_i):
        self.error = y - y_i
        self.delta_i = self.error * self.sigmoid_first(self.a_i)

        self.inner_error = self.delta_i.dot(self.weight_ij.T)
        self.delta_j = self.inner_error*self.sigmoid_first(self.y_j)

        self.weight_ij += learning_rate*self.y_j.T.dot(self.delta_i)
        self.weight_jk += learning_rate*X.T.dot(self.delta_j)


    def learn(self, X, y):
        y_i = self.feed_forward(X)
        self.backprop(X, y, y_i)


learning_rate = 0.2
num_iterations = 1000
model = MultiLayerPerceptron()

for i in range(num_iterations):
    model.learn(X_train, y_train)

print(model.weight_ij)
print(model.weight_jk)