from tensorflow.keras.layers import Layer
import tensorflow as tf


class WrapAroundPadding3D(Layer):
    # Tuple of integers supposed to be symetrical wrapping
    def __init__(self, padding, **kwargs):
        self.padding = padding
        kwargs['trainable'] = False
        super(WrapAroundPadding3D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # Has to be a variable to support reloading
        super(WrapAroundPadding3D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        partial_output_1 = self.wrap_around_data(inputs, 1, self.padding[0])
        partial_output_2 = self.wrap_around_data(partial_output_1, 2, self.padding[1])
        output = self.wrap_around_data(partial_output_2, 3, self.padding[2])

        return output

    def get_config(self):
        return {'padding' : self.padding}

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)

        for axis, wrap in zip(range(1, 4), self.padding):
            output_shape[axis] = output_shape[axis] + 2 * wrap

        return tuple(output_shape)

    def wrap_around_data(self, input, axis, wrap):
        # Right edge
        input_size = tf.shape(input)[axis]

        def wrap_func():
            begin = [0, 0, 0, 0, 0]
            size = [-1, -1, -1, -1, -1]
            size[axis] = wrap
            edge_right = tf.slice(input, begin, size)

            begin[axis] = input_size - wrap
            edge_left = tf.slice(input, begin, size)
            return tf.concat((edge_left, input, edge_right), axis=axis)

        cond = tf.constant(wrap > 0, dtype=tf.bool)
        output = tf.cond(cond, wrap_func, lambda: input)

        return output


def main():
    layer = WrapAroundPadding3D((0, 1, 1))
    tensor = tf.range(0, 9)
    tensor = tf.reshape(tensor, [1, 1, 3, 3, 1])
    padded_tensor = layer.call(tensor)

    print('Tensor')
    print(tensor)

    print('Padded tensor')
    print(padded_tensor)


if __name__ == '__main__':
    main()