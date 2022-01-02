import numpy as np
import neuralNetwork as nn
import datahandler as dh




def main():
    images, labels = dh.load_training_data()
    ni, w, h = images.shape
    images = images.reshape(ni,w*h)


    network = nn.Network([28*28,16,16,10])
    for i in range(0, 12):
        print(i)
        network.backpropdata(images[5000*i:5000*(i+1)],labels[5000*i:5000*(i+1)])
    network.savenetwork("weights.npy","biases.npy")



if __name__ == "__main__":
    main()
