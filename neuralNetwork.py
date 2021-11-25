import numpy as np
import random

class Network:
    def __init__(self, type):
        #type list with number of nodes per layer
        self.type = type
        self.n_layers = len(type)
        self.biases = [np.random.rand(x) for x in type[1:]]
        self.weights = [np.random.rand(x,y) for x, y in zip(type[1:], type[:-1]))]

    def forward(self, input):
        for i in range(self.n_layers-1):
            for weight,bias in zip(self.weights, self.biases):
                input = sigmoid(weight@input+b)
        return input

    def costLabel(self, input, label):
        output = forward(input)
        return np.dot(output-label,output-label)

    #returns average cost given a set of data
    def costData(self, inputs, labels):
        totalCost=0
        for input, label in zip(inputs, labels):
            totalCost+=costLabel(input, label)
        return totalCost/len(inputs)

    def backprop(self, inputs, labels):
        gradw = [np.zeros(np.shape(weight)) for weight in self.weights]
        gradb = [np.zeros(np.shape(bias)) for bias in self.biases]

    def sigmoid(self, argument):
        return 1/(1+np.exp(-argument))
    def sigmoidprime(self, argument):
        exp(-argument)/(1+np.exp(-argument))**2
