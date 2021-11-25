import numpy as np
#import matplotlib.pyplot as plt
import neuralNetwork as nn
import datahandler as dh



def main():
    images, labels = dh.load_training_data()

    network = nn.Network([28*28,16,16,10])
    #print(network.costData(image[0].flatten()))
    print(network.costData(images.flatten(),labels))
main()
