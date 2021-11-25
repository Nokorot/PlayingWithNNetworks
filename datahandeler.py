import numpy as np

TRAINING_DATA = ["data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte"]
TETS_DATA = ["data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte"]

def _read_images_file(filename):
    with open(filename, "rb") as f:
        data = f.read()
    n_of_imgs = int.from_bytes(data[4:8], "big")
    n_of_rows = int.from_bytes(data[8:12], "big")
    n_of_cols = int.from_bytes(data[12:16], "big")

    images = numpy.zeros((n_of_imgs,n_of_rows, n_of_cols), dtype=numpy.uint8)

    for k in range(16, len(data)):
        imgi = k//(n_of_rows*n_of_cols)
        rowi = (k - imgi*n_of_rows*n_of_cols)//n_of_cols
        coli = (k - imgi*n_of_rows*n_of_cols-rowi*n_of_cols)
        images[imgi,rowi,coli] = data[i]
    return images

def _read_labels_file(filename):
    with open(filename, "rb") as f:
        labels = f.read()
    labels = numpy.frombuffer(labels, dtype=numpy.uint8)
    labels = numpy.delete(labels, [0,1,2,3,4,5,6,7])
    return labels

from os.path import exists

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
    return load_data_pairs(TRAINING_DATA)
    
def load_test_data():
    return load_data_pairs(TEST_DATA)

