
mkdir -p data
cd data

# Training data images
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
# Training data labels
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
# Test data images
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
# Test data labels
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

