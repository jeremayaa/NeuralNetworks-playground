import numpy as np
from Network import Network, to_categorical, sigmoid

train_images = np.load('digit_recognition/MNIST preprocessing/train_images.npy')
train_labels = np.load('digit_recognition/MNIST preprocessing/train_labels.npy')
test_images = np.load('digit_recognition/MNIST preprocessing/test_images.npy')
test_labels = np.load('digit_recognition/MNIST preprocessing/test_labels.npy')

train_images = [np.reshape(x, (784, 1)) for x in train_images]
train_labels = [to_categorical(y) for y in train_labels]
training_data = []
for i in zip(train_images, train_labels):
    training_data.append(i)


test_images = [np.reshape(x, (784, 1)) for x in test_images]
test_labels = [to_categorical(y) for y in test_labels]
test_data = []
for i in zip(test_images, test_labels):
    test_data.append(i)

net = Network([784, 30, 10])

net.SGD(training_data, 20, 10, 0.2, test_data)

# about 80% accuracy

# np.save('Network_weights', net.weights, allow_pickle=True)
# np.save('Network_biases', net.biases, allow_pickle=True)

