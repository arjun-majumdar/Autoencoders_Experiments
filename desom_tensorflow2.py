import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
# import seaborn as sns
from tensorflow.keras.layers import AveragePooling2D, Conv2D, MaxPooling2D, ReLU, LeakyReLU
from tensorflow.keras import models, layers, datasets
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputSpec, InputLayer, Activation, Layer
from tensorflow.keras.layers import BatchNormalization, Dropout, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
import numpy as np


print(f"\n\nTF version: {tf.__version__}")

# Check GPU availibility-
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"GPU: {gpu_devices}")

if gpu_devices:
    print(f"GPU: {gpu_devices}")
    details = tf.config.experimental.get_device_details(gpu_devices[0])
    print(f"GPU details: {details.get('device_name', 'Unknown GPU')}")
else:
    print("No GPU found")


# Specify hyper-parameters-
batch_size = 1024
latent_dim = 84

num_classes = 10
num_epochs = 100


# MNIST Data Pre-processing-
# input image dimensions
img_rows, img_cols = 28, 28

# Load MNIST dataset-
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if tf.keras.backend.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print(f"\ninput_shape to be used: {input_shape}\n")

# Convert datasets to floating point types-
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# By default the image data consists of integers between 0 and 255
# for each pixel channel. Neural networks work best when each input
# is inside the range –1 to 1, so we need to divide by 255.

# Normalize the training and testing datasets-
X_train /= 255.0
X_test /= 255.0

# convert class vectors/target to binary class matrices or one-hot encoded values-
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Reshape for dense layers-
X_train = X_train.reshape(X_train.shape[0], 28 * 28 * 1)
X_test = X_test.reshape(X_test.shape[0], 28 * 28 * 1)

# print("\nDimensions of training and testing sets are:")
# print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
# print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")


# Create training and testing datasets-
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

train_dataset = train_dataset.shuffle(buffer_size = 20000, reshuffle_each_iteration = True).batch(batch_size = batch_size, drop_remainder = True)
test_dataset = test_dataset.batch(batch_size = batch_size, drop_remainder = True)


class SOMLayer(Layer):
    """
    Self-Organizing Map layer class with rectangular topology

    # Example
    ```
        model.add(SOMLayer(map_size=(10,10)))
    ```
    # Arguments
        map_size: Tuple representing the size of the rectangular map. Number of prototypes is map_size[0]*map_size[1].
        prototypes: Numpy array with shape `(n_prototypes, latent_dim)` witch represents the initial cluster centers
    # Input shape
        2D tensor with shape: `(n_samples, latent_dim)`
    # Output shapefrom tensorflow.keras.layers import Layer, InputSpec
        2D tensor with shape: `(n_samples, n_prototypes)`
    """

    def __init__(self, map_size, prototypes=None, **kwargs):
        if 'input_shape' not in kwargs and 'latent_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('latent_dim'),)
        super(SOMLayer, self).__init__(**kwargs)
        self.map_size = map_size
        self.n_prototypes = map_size[0]*map_size[1]
        self.initial_prototypes = prototypes
        self.input_spec = InputSpec(ndim=2)
        self.prototypes = None
        self.built = False

    def build(self, input_shape):
        assert(len(input_shape) == 2)
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype = tf.float32, shape = (None, input_dim))
        self.prototypes = self.add_weight(
            shape = (self.n_prototypes, input_dim), initializer = 'glorot_uniform',
            name ='prototypes'
            )

        if self.initial_prototypes is not None:
            self.set_weights(self.initial_prototypes)
            del self.initial_prototypes
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Calculate pairwise squared euclidean distances between inputs and prototype vectors

        Arguments:
            inputs: the variable containing data, Tensor with shape `(n_samples, latent_dim)`
        Return:
            d: distances between inputs and prototypes, Tensor with shape `(n_samples, n_prototypes)`
        """
        # Note: (tf.expand_dims(inputs, axis=1) - self.prototypes) has shape (n_samples, n_prototypes, latent_dim)
        d = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.prototypes), axis=2)
        return d

    def compute_output_shape(self, input_shape):
        assert(input_shape and len(input_shape) == 2)
        return input_shape[0], self.n_prototypes

    def get_config(self):
        config = {'map_size': self.map_size}
        base_config = super(SOMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def som_loss(weights, distances):
    """SOM loss

    Parameters
    ----------
    weights : Tensor, shape = [n_samples, n_prototypes]
        weights for the weighted sum
    distances : Tensor ,shape = [n_samples, n_prototypes]
        pairwise squared euclidean distances between inputs and prototype vectors

    Returns
    -------
    som_loss : loss
        SOM distortion loss
    """
    return tf.reduce_mean(tf.reduce_sum(weights * distances, axis = 1))


# SOM hyper-params-
map_height = 10
map_width = 10

gamma = 0.001

# Total number of train steps/iterations-
total_iterations = len(train_dataset) * num_epochs

# Temperature hyper-parm controlling radius of Gaussian neighborhood-
Tmax = 10.0
Tmin = 0.1


def mlp_autoencoder(
    encoder_dims, act = 'relu',
    init = 'glorot_uniform', batchnorm = False
    ):
    """
    Fully connected symmetric autoencoder model.

    Parameters
    ----------
    encoder_dims : list
        number of units in each layer of encoder. encoder_dims[0] is the input dim, encoder_dims[-1] is the
        size of the hidden layer (latent dim). The autoencoder is symmetric, so the total number of layers
        is 2*len(encoder_dims) - 1
    act : str (default='relu')
        activation of AE intermediate layers, not applied to Input, Hidden and Output layers
    init : str (default='glorot_uniform')
        initialization of AE layers
    batchnorm : bool (default=False)
        use batch normalization

    Returns
    -------
    ae_model, encoder_model, decoder_model : tuple
        autoencoder, encoder and decoder models
    """
    n_stacks = len(encoder_dims) - 1

    # Input
    x = Input(shape=(encoder_dims[0],), name='input')
    # Internal layers in encoder
    encoded = x
    for i in range(n_stacks-1):
        encoded = Dense(encoder_dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(encoded)
        if batchnorm:
            encoded = BatchNormalization()(encoded)
    # Hidden layer (latent space)
    encoded = Dense(encoder_dims[-1], activation='linear', kernel_initializer=init,
                    name='encoder_%d' % (n_stacks - 1))(encoded)  # latent representation is extracted from here
    # Internal layers in decoder
    decoded = encoded
    for i in range(n_stacks-1, 0, -1):
        decoded = Dense(encoder_dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(decoded)
        if batchnorm:
            decoded = BatchNormalization()(decoded)
    # Output
    decoded = Dense(encoder_dims[0], activation='linear', kernel_initializer=init, name='decoder_0')(decoded)

    # AE model
    autoencoder = Model(inputs=x, outputs=decoded, name='AE')

    # Encoder model
    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    # Create input for decoder model
    encoded_input = Input(shape=(encoder_dims[-1],))
    # Internal layers in decoder
    decoded = encoded_input
    for i in range(n_stacks-1, -1, -1):
        decoded = autoencoder.get_layer('decoder_%d' % i)(decoded)
    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder')

    return autoencoder, encoder, decoder


def conv2d_autoencoder(
    input_shape, latent_dim,
    encoder_filters, filter_size,
    pooling_size, act = 'relu',
    init = 'glorot_uniform', batchnorm = False
    ):
    """
    2D convolutional autoencoder model.

    Parameters
    ----------
    input_shape : tuple
        input shape given as (height, width, channels) tuple
    latent_dim : int
        dimension of latent code (units in hidden dense layer)
    encoder_filters : list
        number of filters in each layer of encoder. The autoencoder is symmetric,
        so the total number of layers is 2*len(encoder_filters) - 1
    filter_size : int
        size of conv filters
    pooling_size : int
        size of maxpool filters
    act : str (default='relu')
        activation of AE intermediate layers, not applied to Input, Hidden and Output layers
    init : str (default='glorot_uniform')
        initialization of AE layers
    batchnorm : boolean (default=False)
        use batch normalization

    Returns
    -------
        ae_model, encoder_model, decoder_model : tuple
            autoencoder, encoder and decoder models
    """
    n_stacks = len(encoder_filters)

    # Infer code shape (assuming "same" padding, conv stride equal to 1 and max pooling stride equal to pooling_size)
    code_shape = list(input_shape)
    for _ in range(n_stacks):
        code_shape[0] = int(np.ceil(code_shape[0] / pooling_size))
        code_shape[1] = int(np.ceil(code_shape[1] / pooling_size))
    code_shape[2] = encoder_filters[-1]

    # Input
    x = Input(shape=input_shape, name='input')
    # Internal layers in encoder
    encoded = x
    for i in range(n_stacks):
        encoded = Conv2D(encoder_filters[i], filter_size, activation=act, padding='same', name='encoder_conv_%d'
                                                                                               % i)(encoded)
        if batchnorm:
            encoded = BatchNormalization()(encoded)
        encoded = MaxPooling2D(pooling_size, padding='same', name='encoder_maxpool_%d' % i)(encoded)
    # Flatten
    flattened = Flatten(name='flatten')(encoded)
    # Project using dense layer
    code = Dense(latent_dim, name='dense1')(flattened)  # latent representation is extracted from here
    # Project back to last feature map dimension
    reshaped = Dense(code_shape[0] * code_shape[1] * code_shape[2], name='dense2')(code)
    # Reshape
    reshaped = Reshape(code_shape, name='reshape')(reshaped)
    # Internal layers in decoder
    decoded = reshaped
    for i in range(n_stacks-1, -1, -1):
        if i > 0:
            decoded = Conv2D(encoder_filters[i], filter_size, activation=act, padding='same', name='decoder_conv_%d'
                                                                                                   % i)(decoded)
        else:
            decoded = Conv2D(encoder_filters[i], filter_size, activation=act, padding='valid', name='decoder_conv_%d'
                                                                                                    % i)(decoded)
        if batchnorm:
            decoded = BatchNormalization()(decoded)
        decoded = UpSampling2D(pooling_size, name='decoder_upsample_%d' % i)(decoded)
    # Output
    decoded = Conv2D(1, filter_size, activation='linear', padding='same', name='decoder_0')(decoded)

    # AE model
    autoencoder = Model(inputs=x, outputs=decoded, name='AE')

    # Encoder model (flattened output)
    encoder = Model(inputs=x, outputs=code, name='encoder')

    # Decoder model
    latent_input = Input(shape=(latent_dim,))
    flat_encoded_input = autoencoder.get_layer('dense2')(latent_input)
    encoded_input = autoencoder.get_layer('reshape')(flat_encoded_input)
    decoded = encoded_input
    for i in range(n_stacks-1, -1, -1):
        decoded = autoencoder.get_layer('decoder_conv_%d' % i)(decoded)
        decoded = autoencoder.get_layer('decoder_upsample_%d' % i)(decoded)
    decoded = autoencoder.get_layer('decoder_0')(decoded)
    decoder = Model(inputs=latent_input, outputs=decoded, name='decoder')

    return autoencoder, encoder, decoder




class DESOM(Model):
    def __init__(
        self, map_height = 10,
        map_width = 10, latent_dim = 50,
        encoder_dims = [1, 500, 500, 100]
        ):
        super(DESOM, self).__init__()
        self.map_height = map_height
        self.map_width = map_width
        self.map_size = (self.map_height, self.map_width)
        self.latent_dim = latent_dim
        self.n_prototypes = self.map_size[0] * self.map_size[1]
        self.encoder_dims = encoder_dims
        self.encoder_dims.append(self.latent_dim)

        self.autoencoder, self.encoder, self.decoder = mlp_autoencoder(
            # encoder_dims = [X_train.shape[-1], 500, 500, 2000, latent_dim],
            encoder_dims = self.encoder_dims,
            act = 'relu', init = 'glorot_uniform',
            batchnorm = False
        )

        # Initialize SOM layer-
        self.som_layer = SOMLayer(
            map_size = (self.map_height, self.map_width), name = 'SOM'
        )(self.encoder.output)

        # Create DESOM model
        self.model = Model(
            inputs = self.autoencoder.input,
            outputs = [self.autoencoder.output, self.som_layer]
        )


    def compile(self, gamma:float = 0.001, optimizer:str = 'adam') -> None:
        """
        Compile DESOM model

        Parameters
        ----------
        gamma : float
            coefficient of SOM loss (hyperparameter)
        optimizer : str (default='adam')
            optimization algorithm
        """
        self.model.compile(
            loss = {'decoder_0': 'mse', 'SOM': som_loss},
            # loss_weights = [1, gamma],
            loss_weights = {'decoder_0': 1.0, 'SOM': gamma},
            optimizer = optimizer
        )

        return None


    def predict(self, x):
        """
        Predict best-matching unit using the output of SOM layer

        Parameters
        ----------
        x : array, shape = [n_samples, input_dim] or [n_samples, height, width, channels]
            input samples

        Returns
        -------
        y_pred : array, shape = [n_samples]
            index of the best-matching unit
        """
        _, d = self.model.predict(x, verbose = 0)
        return d.argmin(axis = 1)


    def map_dist(self, y_pred):
        """
        Calculate pairwise Manhattan distances between cluster assignments and map prototypes
        (rectangular grid topology)

        Parameters
        ----------
        y_pred : array, shape = [n_samples]
            cluster assignments

        Returns
        -------
        d : array, shape = [n_samples, n_prototypes]
            pairwise distance matrix on the map
        """
        '''
        labels = np.arange(self.n_prototypes)
        tmp = np.expand_dims(y_pred, axis=1)
        d_row = np.abs(tmp - labels) // self.map_size[1]
        d_col = np.abs(tmp % self.map_size[1] - labels % self.map_size[1])
        return d_row + d_col
        '''
        # y_pred = tf.argmin(input = pairwise_squared_l2dist, axis = 1)
        labels = tf.range(self.n_prototypes)
        tmp = tf.cast(
            x = tf.expand_dims(input = y_pred, axis = 1),
            dtype = tf.dtypes.int32
        )
        # print(labels.dtype, tmp.dtype, y_pred.dtype)
        d_row = tf.abs(tmp - labels) // self.map_size[1]
        d_col = tf.abs(tmp % self.map_size[1] - labels % self.map_size[1])

        # (d_row + d_col).dtype
        # tf.int32

        d_row = tf.cast(x = d_row, dtype = tf.dtypes.float32)
        d_col = tf.cast(x = d_col, dtype = tf.dtypes.float32)

        return d_row + d_col


    def neighborhood_function(
        self, d,
        T, neighborhood = 'gaussian'
    ):
        """
        SOM neighborhood function (Gaussian neighborhood)

        Parameters
        ----------
        d : int
            distance on the map
        T : float
            temperature parameter (neighborhood radius)
        neighborhood : str
            type of neighborhood function ('gaussian' or 'window')

        Returns
        -------
        w : float in [0, 1]
            neighborhood weights
        """
        if neighborhood == 'gaussian':
            # return np.exp(-(d ** 2) / (T ** 2))
            return tf.exp(-tf.square(d) / tf.square(T))
        elif neighborhood == 'window':
            # return (d <= T).astype(np.float32)
            return tf.cast(x = (d <= T), dtype = tf.dtypes.float32)
        else:
            raise ValueError('invalid neighborhood function')


# Initialize MLP AE DESOM model-
model = DESOM(
    map_height = map_height, map_width = map_width,
    latent_dim = latent_dim,
    encoder_dims = [784, 500, 500, 100]
)

# Compile model-
model.compile(gamma = gamma, optimizer = 'adam')

# Required for computing temperature for current train step-
# curr_iter = 1
curr_iter = tf.constant(1)
total_iterations = tf.cast(x = total_iterations, dtype = tf.dtypes.int32)

# Train loss-
train_loss = list()


for epoch in range(1, num_epochs + 1):
    for x, _ in train_dataset:

        # Compute bmu/cluster assignments for batch-
        # _, d = model.model.predict(x)
        _, d = model.model(x)
        # y_pred = d.argmin(axis = 1)
        y_pred = tf.argmin(input = d, axis = 1)
        y_pred = tf.cast(x = y_pred, dtype = tf.dtypes.float32)

        # y_pred.shape, d.shape
        # ((1024,), (1024, 100))

        # Compute temperature for current train step-
        curr_T = tf.cast(
            x = Tmax * tf.pow((Tmin / Tmax), (curr_iter / total_iterations)),
            dtype = tf.dtypes.float32
            )

        # Compute topographic (neighborhood) weights for this batch-
        w_batch = model.neighborhood_function(
            d = model.map_dist(y_pred = y_pred),
            T = curr_T, neighborhood = 'gaussian'
        )

        # Train on batch-
        loss = model.model.train_on_batch(x = x, y = [x, w_batch])
        train_loss.append(loss.item())

        curr_iter += 1


