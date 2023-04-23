import numpy as np
from Network import Network, vectorized_result, sigmoid

train_images = np.load('MNIST preprocessing/train_images.npy')
train_labels = np.load('MNIST preprocessing/train_labels.npy')
test_images = np.load('MNIST preprocessing/test_images.npy')
test_labels = np.load('MNIST preprocessing/test_labels.npy')

train_images = [np.reshape(x, (784, 1)) for x in train_images]
train_labels = [vectorized_result(y) for y in train_labels]
training_data = []
for i in zip(train_images, train_labels):
    training_data.append(i)


test_images = [np.reshape(x, (784, 1)) for x in test_images]
test_labels = [vectorized_result(y) for y in test_labels]
test_data = []
for i in zip(test_images, test_labels):
    test_data.append(i)

net = Network([784, 30, 10])

net.SGD(training_data, 20, 100, 3, test_data)

# np.save('Network_weights', net.weights, allow_pickle=True)
# np.save('Network_biases', net.biases, allow_pickle=True)

