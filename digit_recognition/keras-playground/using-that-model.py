from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# load the trained model
model = load_model("digit_recognition/keras-playground/Keras_model.h5")

# load test images
test_images = np.load('digit_recognition/MNIST preprocessing/test_images.npy')
test_images = test_images.reshape(-1, 784) / 255.0

# load labels 
test_labels = np.load('digit_recognition/MNIST preprocessing/test_labels.npy')


# make predictions on the test images
predictions = model.predict(test_images)


import matplotlib.animation as animation

fig, ax = plt.subplots()

def animate(i):
    ax.clear()  # clear the previous image
    ax.imshow(test_images[i].reshape(28,28), cmap='gray')
    ax.set_title("Label {}, guess {}".format(test_labels[i], np.argmax(predictions[i])))

ani = animation.FuncAnimation(fig, animate, frames=100, interval=700)
plt.show()