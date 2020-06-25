from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow as tf


class MassConversation1D(Layer):
    def __init__(self, volume_data, **kwargs):
        self.volume_data = volume_data
        self.volume_kernel = tf.Variable(self.volume_data, trainable=False, dtype=tf.float32)
        kwargs['trainable'] = False
        super(MassConversation1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # Has to be a variable to support reloading
        super(MassConversation1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        input, output = inputs

        # Produces vectors with
        # input_mass[i] = mass of sample i in batch
        # outpus_mass[i] = mass of output for sample i in batch
        input_mass = input * self.volume_kernel
        input_mass = K.sum(input_mass, [1, 2])

        output_mass = output * self.volume_kernel
        output_mass = K.sum(output_mass, [1, 2])

        # Together with the broadcasting of the vector this produces a matrix with
        # i,_,_,_,_ = (input_mass[i]/outpus_mass[i])
        # Multiplication with this matrix scales the output for each sample so that the proportions are kept
        # and the output mass is the same as the input mass for each sample in the batch
        vector = (input_mass / output_mass)[np.newaxis][np.newaxis]
        vector = K.transpose(vector)
        normalized_output = output * vector

        return normalized_output

    def get_config(self):
        return {'volume_data': self.volume_data}

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape1, shape2 = input_shape

        assert shape1 == shape2  # else normalization is not possible
        return [tuple(shape2)]
