import numpy as np
import neuralNetwork as nn
import datahandler as dh




def main():
    images, labels = dh.load_training_data()
    ni, w, h = images.shape
    images = images.reshape(ni,w*h)


    network = nn.Network([28*28,16,16,10])
    network.backprop(images[0],labels[0])


if __name__ == "__main__":
    main()
