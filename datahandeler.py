import numpy as np
from os.path import exists

TRAINING_DATA = ["data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte"]
TETS_DATA = ["data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte"]

def _read_images_file(filename):
    print(f"Reading image file `{filename}`!")

    with open(filename, "rb") as f:
        data = f.read()
    n_of_imgs = int.from_bytes(data[4:8], "big")
    n_of_rows = int.from_bytes(data[8:12], "big")
    n_of_cols = int.from_bytes(data[12:16], "big")

    images = np.zeros((n_of_imgs,n_of_rows, n_of_cols), dtype=np.uint8)

    for (k,byte) in enumerate(data[16:]):
        imgi = k//(n_of_rows*n_of_cols)
        rowi = (k - imgi*n_of_rows*n_of_cols)//n_of_cols
        coli = (k - imgi*n_of_rows*n_of_cols-rowi*n_of_cols)
        
        if k%(len(data)//10) == 0:
            print(f"{k//(len(data)//10)}0%")
        images[imgi,rowi,coli] = byte; 
    return images

def _read_labels_file(filename):
    print(f"Reading label file `{filename}`!")
    with open(filename, "rb") as f:
        labels = f.read()
    data = np.frombuffer(labels[8:], dtype=np.uint8)
    labels = np.zeros((10, len(labels)), dtype=np.uint8)
    
    for (i,k) in enumerate(data):
        labels[k,i] = 1;

    print(data, labels)

    return labels

def _load_data(filepath, reader):
    npy_filepath = f"{filepath}.npy"
    if exists(npy_filepath):
        return np.load(npy_filepath);
    data = reader(filepath)
    np.save(npy_filepath, data)
    return data 


def load_data_pair(filepaths):
    imgs = _load_data(filepaths[0], _read_images_file)
    labels = _load_data(filepaths[1], _read_labels_file)
    return imgs, labels

def load_training_data():
    return load_data_pair(TRAINING_DATA)
    
def load_test_data():
    return load_data_pair(TEST_DATA)

