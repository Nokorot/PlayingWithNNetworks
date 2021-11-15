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

#readImg("data/train-images-idx3-ubyte")
data=numpy.load("images.npy")
plt.imshow(data[0])
plt.show()
