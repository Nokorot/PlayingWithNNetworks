import numpy as np
import random

def sigmoid(argument):
    return 1/(1+np.exp(-argument))
def sigmoidprime(argument):
    return np.exp(-argument)/(1+np.exp(-argument))**2

class Network:
    def __init__(self, type, weights = None, biases = None):
        #type list with number of nodes per layer
        self.type = type
        self.n_layers = len(type)
        if(biases == None):
            self.biases = [np.random.rand(x,1) for x in type[1:]]
        else:
            self.biases = biases
        if(weights == None):
            self.weights = [np.random.rand(x,y) for x, y in zip(type[1:], type[:-1])]
        else:
            self.weights = weights

    def loadfromfile(type,filepath1,filepath2):
        w = []
        b = []
        wflat = np.load(filepath1)
        bflat = np.load(filepath2)
        wstartpoint = 0
        bstartpoint = 0
        for i in range (len(type)-1):
            wsize = type[i]*type[i+1]
            w.append(np.reshape(wflat[wstartpoint:wstartpoint+wsize],(type[i+1],type[i])))
            wstartpoint += wsize

            bsize = type[i+1]
            b.append(bflat[bstartpoint:bstartpoint+type[i+1]])
            bstartpoint += bsize
        return Network(type, w, b )
    #    f1=np.load(filepath1)
    #    f2=np.load(filepath2)
    #    biases = [for i in type[1:]]
    #    biases.append()


    def forward(self, input):
        input = input.reshape(len(input),1)
        for weight,bias in zip(self.weights, self.biases):
            input = sigmoid(weight@input+bias)
        return input

    def costLabel(self, input, label):
        output = self.forward(input)
        label = np.reshape(len(input),1)
        return np.dot(np.transpose(output-label),output-label)

    #returns average cost given a set of data
    def costData(self, inputs, labels):
        totalCost=0
        for input, label in zip(inputs, labels):
            totalCost+=self.costLabel(input, label)
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
            z = weight@input + bias
            input = sigmoid(z)
            zs.append(z)
            layers.append(input)

        multiplier = 2*(layers[-1]-label)*sigmoidprime(zs[-1])
        gradw[-1] += multiplier@np.transpose(layers[-2])
        gradb[-1] += multiplier

        #print(sigmoidprime(zs[-1]),zs[-1])

        """2(aj-yj)sigmaprime(z)*ak=delwjk=dC/dwjk"""

        for i in range(2, self.n_layers):
            gradb[-i] += (np.transpose(self.weights[-i+1])@multiplier)*sigmoidprime(zs[-i])
            gradw[-i] += (np.transpose(self.weights[-i+1])@multiplier)*sigmoidprime(zs[-i])@np.transpose(layers[-i-1])
            multiplier = gradb[-i]
        return gradw, gradb

    #updates w, b based on inputs, labels
    def backpropdata(self, inputs, labels):
        gradw = [np.zeros(np.shape(weight)) for weight in self.weights]
        gradb = [np.zeros(np.shape(bias)) for bias in self.biases]
        for input, label in zip(inputs,labels):
            for i, (w, b) in enumerate(zip(*self.backprop(input,label))):
                gradw[i]+=w/len(inputs)
                gradb[i]+=b/len(inputs)

        for i, grad in enumerate(gradw):
            self.weights[i]-=grad
        for i, grad in enumerate(gradb):
            self.biases[i]-=grad

    def savenetwork(self, path1,path2):
        np.save(path1, np.concatenate([w.flatten() for w in self.weights]))
        np.save(path2, np.concatenate([b.flatten() for b in self.biases]))


    def loadnetwork(self, path1, path2):


        return w, b
