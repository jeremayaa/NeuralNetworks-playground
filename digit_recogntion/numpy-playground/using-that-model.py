import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Network import Network, to_categorical


# load the model
biases = np.load('numpy-playground/Network_biases.npy', allow_pickle=True)
weights = np.load('numpy-playground/Network_weights.npy', allow_pickle=True)

# initialize model
net = Network([784, 30, 10])
net.biases = biases
net.weights = weights

# load test data
test_images = np.load('MNIST preprocessing/test_images.npy')
test_input = [np.reshape(x, (784, 1)) for x in test_images]
test_labels = np.load('MNIST preprocessing/test_labels.npy')

# evaluate model and print accuracy
test_data = []
for i in zip([np.reshape(x, (784, 1)) for x in test_images], [to_categorical(y) for y in test_labels]):
    test_data.append(i)
print("Model accuracy: ", net.evaluate(test_data))


# show images with labels and predictions
fig, ax = plt.subplots()

def animate(i):
    ax.clear()
    ax.imshow(test_images[i], cmap='gray')
    a = net.feedforward(test_input[i])
    ax.set_title("number {0}, prediction {1}".format(test_labels[i], np.argmax(a)))
    
ani = animation.FuncAnimation(fig, animate, frames=100, interval=1000)
plt.show()

