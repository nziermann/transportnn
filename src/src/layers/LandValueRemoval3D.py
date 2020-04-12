from tensorflow.keras.layers import Layer
import tensorflow as tf


class LandValueRemoval3D(Layer):
    def __init__(self, land_data, **kwargs):
        self.land_data = land_data
        self.land_kernel = tf.Variable(self.land_data, trainable=False, dtype=tf.float32)
        kwargs['trainable'] = False
        super(LandValueRemoval3D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # Has to be a variable to support reloading
        super(LandValueRemoval3D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, input, **kwargs):
        # Together with the broadcasting of the vector this produces a matrix with
        # _,x,y,z,_ = land_data[x,y,z]
        # Multiplication with this matrix scales the output for each sample so that the proportions are kept
        # and the output mass is the same as the input mass for each sample in the batch
        land_removal_output = input * self.land_kernel

        return land_removal_output

    def get_config(self):
        return {'land_data': self.land_data}