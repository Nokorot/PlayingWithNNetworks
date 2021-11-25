import numpy as np
import random

class Network:
    def __init__(self, type):
        #type list with number of nodes per layer
        self.type = type
        self.n_layers = len(type)
        self.biases = [np.random.rand(x,1) for x in type[1:]]
        self.weights = [np.random.rand(x,y) for x, y in zip(type[1:], type[:-1])]

    def forward(self, input):
        input = input.reshape(1,len(input))
        for i in range(self.n_layers-1):
            for weight,bias in zip(self.weights, self.biases):
                input = self.sigmoid(weight@input+b)
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

    #backprop for 1 input-label
    def backprop(self, input, label):
        input = input.reshape(len(input),1)
        label = label.reshape(len(label),1)
        #list of grads in the for of weights and biases from self:
        gradw = [np.zeros(np.shape(weight)) for weight in self.weights]
        gradb = [np.zeros(np.shape(bias)) for bias in self.biases]
        #do the forward propagation, saving everything:
        layers = [input]
        zs = []

        for weight, bias in zip(self.weights, self.biases):
            input = weight@input + bias
            zs.append(input)
            layers.append(sigmoid(input))

        gradw[0][j,k] = 2*(layers[-1]-label)*sigmoidprime(zs[-1])@np.transpose(layers[-2])
        gradb[0] = 2*(layers[-1]-label)*sigmoidprime(zs[-1])

            """2(aj-yj)sigmaprime(z)*ak=delwjk=dC/dwjk"""


    def sigmoid(self, argument):
        return 1/(1+np.exp(-argument))
    def sigmoidprime(self, argument):
        exp(-argument)/(1+np.exp(-argument))**2
