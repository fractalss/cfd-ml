# `ml_rom/utils/layers.py`

from tensorflow.keras.layers import Layer
import tensorflow as tf


class ExpandDimsLayer(Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=1)


class TileLayer(Layer):
    def __init__(self, n_points, **kwargs):
        super().__init__(**kwargs)
        self.n_points = n_points

    def call(self, inputs):
        return tf.tile(inputs, [1, self.n_points, 1])