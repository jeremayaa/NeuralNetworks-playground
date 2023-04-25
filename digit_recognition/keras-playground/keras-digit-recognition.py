from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import backend as K


import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
train_images = np.load('digit_recognition/MNIST preprocessing/train_images.npy')
train_labels = np.load('digit_recognition/MNIST preprocessing/train_labels.npy')
test_images = np.load('digit_recognition/MNIST preprocessing/test_images.npy')
test_labels = np.load('digit_recognition/MNIST preprocessing/test_labels.npy')

# Neurons take values from 0 to 1 as an input
train_images = train_images.reshape(-1, 784) / 255.0
test_images = test_images.reshape(-1, 784) / 255.0

# labels are integers 0-9. 
# 'to_categorical' makes an array of length 10 for each number
# This array is all zeros with a 'one' in the index coresponding to the given number
# f.e: 3 -> [0 0 0 1 0 0 0 0 0 0]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Create the neural network
model = Sequential()

# Create custom activaton function
# sin(x)^2 that i've try for fun, turn out to have better percofmence than sigmoid
def square_sin_activation(x):
    return K.square(K.sin(x))


# model.add(Dense(30, activation=square_sin_activation, input_shape=(784,)))
# model.add(Dense(30, activation='relu', input_shape=(784,)))
model.add(Dense(30, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation="sigmoid"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")


# model.save('Keras_model.h5')