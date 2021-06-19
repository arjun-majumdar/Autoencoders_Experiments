

"""
Autoencoders in TensorFlow2

An autoencoder is a special type of neural network that is trained to copy its input to its output. For
example, given an image of a handwritten digit, an autoencoder first encodes the image into a lower
dimensional latent representation, then decodes the latent representation back to an image. An autoencoder
learns to compress the data while minimizing the reconstruction error.

To learn more about autoencoders, please consider reading chapter 14 from Deep Learning by Ian Goodfellow,
Yoshua Bengio, and Aaron Courville using the URL-
https://www.deeplearningbook.org/


Refer-
https://www.tensorflow.org/tutorials/generative/autoencoder#define_a_convolutional_autoencoder
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


print(f"TensorFlow version: {tf.__version__}")
# TensorFlow version: 2.5.0

num_gpus = len(tf.config.list_physical_devices('GPU'))
print(f"number of GPUs available = {num_gpus}")
# number of GPUs available = 1


# Load the dataset
# Fashon MNIST dataset is used. Each image in this dataset is 28x28 pixels (grey-scale images).
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print (f"X_train.shape: {X_train.shape} & X_test.shape: {X_test.shape}")
# X_train.shape: (60000, 28, 28, 1) & X_test.shape: (10000, 28, 28, 1)


# Define an autoencoder with: an encoder, which compresses the images into a d-dimensional latent vector,
# and a decoder, that reconstructs the original image from the latent space.

class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
    
        self.encoder = tf.keras.Sequential([
            # layers.InputLayer(shape = (28, 28, 1)),
            layers.InputLayer(input_shape = (28, 28, 1)),
            layers.Conv2D(
                filters = 16, kernel_size = (3, 3),
                activation='relu', padding = 'same',
                strides = 2),
            layers.Conv2D(
                filters = 8, kernel_size = (3, 3),
                activation = 'relu', padding = 'same',
                strides = 2)
            ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(
                filters = 8, kernel_size = 3,
                strides = 2, activation = 'relu',
                padding = 'same'),
            layers.Conv2DTranspose(
                filters = 16, kernel_size = 3,
                strides = 2, activation = 'relu',
                padding = 'same'),
            layers.Conv2D(
                filters = 1, kernel_size = (3, 3),
                activation = 'sigmoid', padding = 'same')
            ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Initialize an instance of Autoencoder-
autoencoder = Denoise()

# Build autoencoder-
# autoencoder.build(input_shape = (28, 28, 1))

autoencoder.summary()
"""
Model: "denoise"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
sequential (Sequential)      (None, 7, 7, 8)           1320
_________________________________________________________________
sequential_1 (Sequential)    (None, 28, 28, 1)         1897
=================================================================
Total params: 3,217
Trainable params: 3,217
Non-trainable params: 0
_________________________________________________________________
"""

# Count number of trainable parameters manually-
tot_params = 0

for layer in autoencoder.trainable_weights:
    loc_params = tf.math.count_nonzero(layer, axis = None).numpy()
    tot_params += loc_params
    print(f"layer: {layer.shape} has {loc_params} weights")

"""
layer: (3, 3, 1, 16) has 144 weights
layer: (16,) has 0 weights
layer: (3, 3, 16, 8) has 1152 weights
layer: (8,) has 0 weights
layer: (3, 3, 8, 8) has 576 weights
layer: (8,) has 0 weights
layer: (3, 3, 16, 8) has 1152 weights
layer: (16,) has 0 weights
layer: (3, 3, 16, 1) has 144 weights
layer: (1,) has 0 weights
"""

print(f"\nTotal number of trainable weights = {tot_params}")
# Total number of trainable weights = 3168

# Compile defined autoencoder-
autoencoder.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    loss = losses.MeanSquaredError()
    )


# Train the model using X_train as both the input and the target. The encoder will learn to compress the
# dataset from 784 (28 x 28) dimensions to the latent space, and the decoder will learn to reconstruct the
# original images.

history = autoencoder.fit(
    x = X_train, y = X_train,
    epochs = 10, shuffle = True,
    validation_data=(X_test, X_test)
    )


# Look at the summary of encoder-
autoencoder.encoder.summary()
"""
Model: "sequential_16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_24 (Conv2D)           (None, 14, 14, 16)        160
_________________________________________________________________
conv2d_25 (Conv2D)           (None, 7, 7, 8)           1160
=================================================================
Total params: 1,320
Trainable params: 1,320
Non-trainable params: 0
_________________________________________________________________
"""


# Decoder upsamples latent space representations from (7, 7, 8) to (28, 28, 1)-
autoencoder.decoder.summary()
"""
Model: "sequential_17"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_transpose_16 (Conv2DT (None, 14, 14, 8)         584
_________________________________________________________________
conv2d_transpose_17 (Conv2DT (None, 28, 28, 16)        1168
_________________________________________________________________
conv2d_26 (Conv2D)           (None, 28, 28, 1)         145
=================================================================
Total params: 1,897
Trainable params: 1,897
Non-trainable params: 0
_________________________________________________________________
"""


# Visualize images produced by encoder-
encoded_imgs = autoencoder.encoder(X_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

encoded_imgs.shape
# (10000, 7, 7, 8)

decoded_imgs.shape
# (10000, 28, 28, 1)


n = 10

plt.figure(figsize = (20, 4))
for i in range(n):

    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original")
    plt.imshow(tf.squeeze(X_test[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()




"""
Using Autoencoder for Image Denoising:

An autoencoder can also be trained to remove noise from images. In this example a noisy version of the Fashion
MNIST dataset is created by applying random noise to each image. Then, an autoencoder is trained using the
noisy image as input, and the original image as the target.
"""
# Adding random noise to the images-
noise_factor = 0.2

X_train_noisy = X_train + noise_factor * tf.random.normal(shape = X_train.shape) 
X_test_noisy = X_test + noise_factor * tf.random.normal(shape = X_test.shape) 

X_train_noisy.shape, X_test_noisy.shape
# (TensorShape([60000, 28, 28, 1]), TensorShape([10000, 28, 28, 1]))

X_train_noisy = tf.clip_by_value(X_train_noisy, clip_value_min = 0., clip_value_max = 1.)
X_test_noisy = tf.clip_by_value(X_test_noisy, clip_value_min = 0., clip_value_max = 1.)

X_train_noisy.shape, X_test_noisy.shape
# (TensorShape([60000, 28, 28, 1]), TensorShape([10000, 28, 28, 1]))


# Plot the noisy images-
n = 10
plt.figure(figsize = (20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(X_test_noisy[i]))
    plt.gray()
plt.show()


# Initialize an Autoencoder for image denoising-
autoencoder_denoise = Denoise()

# Compile defined autoencoder-
autoencoder_denoise.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    loss = losses.MeanSquaredError()
    )


# Train the model using X_train as both the input and the target. The encoder will learn to compress the
# dataset from 784 (28 x 28) dimensions to the latent space, and the decoder will learn to reconstruct the
# original images.

history_denoise = autoencoder_denoise.fit(
    x = X_train_noisy, y = X_train,
    epochs = 10, shuffle = True,
    validation_data=(X_test_noisy, X_test)
    )
    

# Plotting both the noisy images and the denoised images produced by the autoencoder-
encoded_imgs_denoise = autoencoder_denoise.encoder(X_test).numpy()
decoded_imgs_denoise = autoencoder_denoise.decoder(encoded_imgs_denoise).numpy()


n = 10
plt.figure(figsize = (20, 4))
for i in range(n):

    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(X_test_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs_denoise[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()


# Save trained autoencoders for later use-
autoencoder.save_weights("autoencoder_trained_fashion_mnist.h5", overwrite = True)
autoencoder_denoise.save_weights("autoencoder_denoise_trained_fashion_mnist.h5", overwrite = True)


