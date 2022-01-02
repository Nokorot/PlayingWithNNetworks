import numpy as np
import neuralNetwork as nn
import datahandler as dh




def train():
    images, labels = dh.load_training_data()
    ni, w, h = images.shape
    images = images.reshape(ni,w*h)/255


    network = nn.Network([28*28,16,16,10])
    print(network.costData(images, labels))
    for i in range(0, 20):
        print(i)
        network.backpropdata(images[5000*i:5000*(i+1)],labels[5000*i:5000*(i+1)])
    print(network.costData(images,labels))
    network.savenetwork("weights.npy","biases.npy")


def main():
    train()
    network = nn.Network.loadfromfile([28*28, 16,16,10],"weights.npy", "biases.npy")


if __name__ == "__main__":
    main()
