import numpy
import matplotlib.pyplot as plt

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

def cost(result, label):
    np.dot(result - label,result-label)

def transformm(vector, weights, biases):
    x = weights * vector + biases
    x = 1/(1+np.exp(-x))

#readImg("data/train-images-idx3-ubyte")
#readLabels("data/train-labels-idx1-ubyte")
data = numpy.load("images.npy")
labels = numpy.load("labels.npy")

imgsize=len(data[0].flatten())

weights1 = np.random.rand((16,imgsize))
weights2 = np.random.rand(16,16)
weights3 = np.random.rand(10,16)
biases1 = np.random.rand(16)
biases2 = np.random.rand(16)
biases3 = np.random.rand(10)

img = images[0].flatten()
def adapt(lbl):
    image = img.flatten()
    label = numpy.zeros(10)
    for i in range(0, 10):
        if labels[0] == i:
            label[i] = 1
    return label

def actualCost(img, label):
    img = transformm(img, weights1, biases1)
    img = transformm(img, weights2, biases2)
    img = transformm(img, weights3, biases3)
    cost = cost(img, label)

#plt.imshow(data[0])
#plt.show()
