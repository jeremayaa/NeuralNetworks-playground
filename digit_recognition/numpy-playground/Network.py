import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
        This initialize weights and biases for network. Sizes is a list containing
        number of neurons in each layer
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, i, get_all=False):
        """
        Return the output of the network if i is input.
        If get all is True, return list of activations in each layer, as
        well as the "zs" which are activations befor applying activation
        """
        if get_all == True: 
            zs = []
            activations = [i]
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, i)+b
                zs.append(z)
                i = sigmoid(z)
                activations.append(i)
            
            return activations, zs
        else:
            for b, w in zip(self.biases, self.weights):
                i = sigmoid(np.dot(w, i)+b)
            return i

    
    def backprop(self, x, y):
        """Return a tuple '(nabla_b, nabla_w)' representing the
        gradient for the cost function C_x. They
        are layer-by-layer lists of numpy arrays, similar
        with the same shape as biases and weights.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activations, zs = self.feedforward(x, get_all=True)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def SGD(self, training_data, epochs, batch_size, learning_rate,
            test_data):
        """
        Implement stochastic gradient descent, by deviding training data
        into random packages (called batches, "batch_size" is number of elements in each
        batch), then it trains network on every batch.
        Process is repeted for every epoch (where epoch refers to all batches in dataset), 
        every epoch is different since every time batches are assembling randomly.

        If 'test_data' is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out."""

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [
                training_data[k:k+batch_size]
                for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, learning_rate)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch {0} complete".format(j))

    def update_batch(self, batch, learning_rate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a batch.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(learning_rate/len(batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]


    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result.
        """
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives - dC/da 
        for the output activations.
        In the case of quadracic cost function, when
        C = 1/2 * sum((y - a)^2), the derivative is simply: a-y
        """
        return (output_activations-y)


#Activation functions
def sigmoid(z):
    """The sigmoid function."""
    # return 1.0/(1.0+np.exp(-z))
    return 0.5 * (1 + np.tanh(0.5*z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def sin_sq(z):
    return np.sin(z)**2

def sin_sq_prime(z):
    return np.sin(2*z)

def softmax(z):
    """The softmax function."""
    z_max = np.max(z)
    e_z = np.exp(z - z_max)
    return e_z / np.sum(e_z)

def softmax_prime(z):
    """Derivative of the softmax function."""
    p = softmax(z)
    return p * (1 - p)

def to_categorical(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network.
    Equivalent of keras function 'to_categorical'
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


