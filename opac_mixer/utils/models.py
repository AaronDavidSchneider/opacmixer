"""Module that houses several of the models that have been tested for ktable mixing emulation"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_simple_1d_conv(ng, num_species):
    """
    Simple CNN model
    warning: fixed number of species

    Parameters
    ----------
    ng: int
        number of $g$ values
    num_species: int
        number of species to be mixed (fixed)

    Returns
    -------
    model: keras.Model
        a keras model
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(ng, num_species)))
    model.add(layers.Permute((2, 1), input_shape=(ng, num_species)))
    model.add(
        layers.Conv1D(
            filters=4, kernel_size=3, activation="linear", padding="same"
        )
    )
    model.add(
        layers.Conv1D(
            filters=4, kernel_size=3, activation="linear", padding="same"
        )
    )
    model.add(
        layers.Conv1D(
            filters=1, kernel_size=3, activation="linear", padding="same"
        )
    )
    model.add(layers.Flatten())
    model.add(layers.Dense(units=ng, activation="linear"))
    return model


def get_simple_mlp(ng, hidden_units=None):
    """
    Simple MLP model with several dense layers
    warning: not permutation invariant!

    Parameters
    ----------
    ng: int
        number of $g$ values
    hidden_units: int
        number of units in the dense layers

    Returns
    -------
    model: keras.Model
        a keras model
    """
    if hidden_units is None:
        hidden_units = ng

    model = keras.Sequential()
    model.add(layers.Input(shape=(ng, None)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=hidden_units, activation="relu"))
    model.add(layers.Dense(units=hidden_units, activation="relu"))
    model.add(layers.Dense(units=hidden_units, activation="relu"))
    model.add(layers.Dense(units=hidden_units, activation="linear"))
    return model


def get_deepset(ng, hidden_units=None):
    """
    DeepSet with MLP as decoder and encoder.
    From the paper

    Parameters
    ----------
    ng: int
        number of $g$ values
    hidden_units: int
        number of units in the encoding dense layer

    Returns
    -------
    model: keras.Model
        a keras model
    """
    if hidden_units is None:
        hidden_units = ng

    model = keras.Sequential(
        [
            layers.Input(shape=(ng, None)),
            layers.Permute((2, 1), input_shape=(ng, None)),
            layers.Dense(
                units=hidden_units, activation="relu", use_bias=False
            ),
            layers.Lambda(lambda x: tf.reduce_sum(x, axis=-2)),
            layers.Dense(units=ng, activation="linear", use_bias=False),
        ]
    )
    return model


def get_unet_1d(ng, species, min_filters=8):
    """
    A U-NET like architecture
    Idea: convolutes the species with each other by first downconvoluting and then upconvoluting

    warning: fixed number of species

    Parameters
    ----------
    ng: int
        number of $g$ values
    species: int
        number of species to be mixed
    min_filters: int
        number of filters in the first convolution layer

    Returns
    -------
    model: keras.Model
        a keras model
    """

    input_layer = layers.Input(shape=(ng, species))

    conv1 = layers.Conv1D(min_filters, 3, activation="relu", padding="same")(
        input_layer
    )
    conv1 = layers.Conv1D(min_filters, 3, activation="relu", padding="same")(
        conv1
    )
    pool1 = layers.MaxPooling1D(2)(conv1)

    conv2 = layers.Conv1D(
        2 * min_filters, 3, activation="relu", padding="same"
    )(pool1)
    conv2 = layers.Conv1D(
        2 * min_filters, 3, activation="relu", padding="same"
    )(conv2)
    pool2 = layers.MaxPooling1D(2)(conv2)

    conv3 = layers.Conv1D(
        4 * min_filters, 3, activation="relu", padding="same"
    )(pool2)
    conv3 = layers.Conv1D(
        4 * min_filters, 3, activation="relu", padding="same"
    )(conv3)

    up4 = layers.Conv1DTranspose(
        2 * min_filters, 2, strides=2, padding="same"
    )(conv3)
    concat4 = layers.concatenate([conv2, up4], axis=-1)
    conv4 = layers.Conv1DTranspose(
        2 * min_filters, 3, activation="relu", padding="same"
    )(concat4)
    conv4 = layers.Conv1DTranspose(
        2 * min_filters, 3, activation="relu", padding="same"
    )(conv4)

    up5 = layers.Conv1DTranspose(min_filters, 2, strides=2, padding="same")(
        conv4
    )
    concat5 = layers.concatenate([conv1, up5], axis=-1)
    conv5 = layers.Conv1DTranspose(
        min_filters, 3, activation="relu", padding="same"
    )(concat5)
    conv5 = layers.Conv1DTranspose(
        min_filters, 3, activation="relu", padding="same"
    )(conv5)

    output_layer = layers.Conv1D(1, 3, activation="linear", padding="same")(
        conv5
    )
    output_layer = layers.Flatten()(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model
