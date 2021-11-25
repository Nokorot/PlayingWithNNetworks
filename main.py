import numpy
#import matplotlib.pyplot as plt
import neuralNetwork.py

def readImg (filename):
    with open(filename, "rb") as f:
        data = f.read()
    n_of_imgs = int.from_bytes(data[4:8], "big")
    n_of_rows = int.from_bytes(data[8:12], "big")
    n_of_cols = int.from_bytes(data[12:16], "big")

    images = numpy.zeros((n_of_imgs,n_of_rows, n_of_cols), dtype=numpy.uint8)

    k = 0
    for i in data[16:]:

        imgi = k//(n_of_rows*n_of_cols)
        rowi = (k - imgi*n_of_rows*n_of_cols)//n_of_cols
        coli = (k - imgi*n_of_rows*n_of_cols-rowi*n_of_cols)
        images[imgi,rowi,coli] = i
        k+=1
        if k%100000 == 0:
            print(k)

    numpy.save("images",images)

def readLabels (filename):
    f = open(filename, "rb")
    labels = f.read()
    labels = numpy.frombuffer(labels, dtype=numpy.uint8)
    labels = numpy.delete(labels, [0,1,2,3,4,5,6,7])
    numpy.save("labels",labels)

def main():
    network = Network([28*28,16,16,10])
