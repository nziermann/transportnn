from keras import backend as K
from keras.layers import Layer
import numpy as np


class LandValueRemoval3D(Layer):
    def __init__(self, land_data, **kwargs):
        self.land_data = land_data
        super(LandValueRemoval3D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # Has to be a variable to support reloading
        self.land_kernel = K.variable(self.land_data)
        self.non_trainable_weights = [self.land_kernel]
        super(LandValueRemoval3D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, input, **kwargs):
        # Together with the broadcasting of the vector this produces a matrix with
        # _,x,y,z,_ = land_data[x,y,z]
        # Multiplication with this matrix scales the output for each sample so that the proportions are kept
        # and the output mass is the same as the input mass for each sample in the batch
        land_removal_output = input * self.land_kernel

        return [land_removal_output]

    def get_config(self):
        config = {'land_data': self.land_data}
        base_config = super(LandValueRemoval3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape