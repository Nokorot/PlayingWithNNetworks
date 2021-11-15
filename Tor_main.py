
import numpy as np
import cv2

def read_img_file(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    magic_num = int.from_bytes(data[0:4], "big");
    size = int.from_bytes(data[4:8], "big");
    row = int.from_bytes(data[8:12], "big");
    col = int.from_bytes(data[12:16], "big");

    imgs = np.zeros((size, col, row), dtype=np.uint8);
    # k = 16;
    for i in range(size):
        for r in range(row):
            for c in range(col):
                imgs[i,r,c] = data[k]
                k += 1;
    return imgs

def read_data_files():
    imgs = read_img_file("data/train-images-idx3-ubyte")
    np.save("train-images-idx3-ubyte", imgs)

def load_data():
    return np.load("train-images-idx3-ubyte.npy");

import matplotlib.pyplot as plt


def main():
    imgs = load_data()
    print(imgs[0].shape)
    plt.imshow(imgs[4])
    plt.show()


# read_data_files()

main()

