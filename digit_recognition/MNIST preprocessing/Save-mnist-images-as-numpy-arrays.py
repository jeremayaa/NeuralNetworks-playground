import gzip
import numpy as np

# Load the MNIST dataset
path = 'digit_recognition/MNIST preprocessing/mnist-dataset/'
with gzip.open(path + 't10k-images-idx3-ubyte.gz', "rb") as f:
    test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

with gzip.open(path + 't10k-labels-idx1-ubyte.gz', "rb") as f:
    test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

with gzip.open(path + 'train-images-idx3-ubyte.gz', "rb") as f:
    train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

with gzip.open(path + 'train-labels-idx1-ubyte.gz', "rb") as f:
    train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

np.save('digit_recognition/MNIST preprocessing/train_images.npy', train_images)
np.save('digit_recognition/MNIST preprocessing/train_labels.npy', train_labels)
np.save('digit_recognition/MNIST preprocessing/test_images.npy', test_images)
np.save('digit_recognition/MNIST preprocessing/test_labels.npy', test_labels)



