# -*- coding: utf-8 -*-
"""Locally-connected layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend as K


class LocallyConnected3D(Layer):
    """Locally-connected layer for 2D inputs.

    The `LocallyConnected2D` layer works similarly
    to the `Conv2D` layer, except that weights are unshared,
    that is, a different set of filters is applied at each
    different patch of the input.

    # Examples
    ```python
        # apply a 3x3 unshared weights convolution with 64 output filters
        # on a 32x32 image with `data_format="channels_last"`:
        model = Sequential()
        model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
        # now model.output_shape == (None, 30, 30, 64)
        # notice that this layer will consume (30*30)*(3*3*3*64)
        # + (30*30)*64 parameters

        # add a 3x3 unshared weights convolution on top, with 32 output filters:
        model.add(LocallyConnected2D(32, (3, 3)))
        # now model.output_shape == (None, 28, 28, 32)
    ```

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        padding: Currently only support `"valid"` (case-insensitive).
            `"same"` will be supported in future.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            tensorflow.keras config file at `~/.tensorflow.keras/tensorflow.keras.json`.
            If you never set it, then it will be "channels_last".
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(LocallyConnected3D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 3, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if self.padding != 'valid':
            raise ValueError('Invalid border mode for LocallyConnected3D '
                             '(only "valid" is supported): ' + padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=5)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if self.data_format == 'channels_last':
            input_row, input_col, input_stack = input_shape[1:-1]
            input_filter = input_shape[-1]
        else:
            input_row, input_col, input_stack = input_shape[2:]
            input_filter = input_shape[1]
        if input_row is None or input_col is None or input_stack is None:
            raise ValueError('The spatial dimensions of the inputs to '
                             ' a LocallyConnected3D layer '
                             'should be fully-defined, but layer received '
                             'the inputs shape ' + str(input_shape))
        output_row = conv_utils.conv_output_length(input_row, self.kernel_size[0],
                                                   self.padding, self.strides[0])
        output_col = conv_utils.conv_output_length(input_col, self.kernel_size[1],
                                                   self.padding, self.strides[1])
        output_stack = conv_utils.conv_output_length(input_stack, self.kernel_size[2],
                                                     self.padding, self.strides[2])

        self.output_row = output_row
        self.output_col = output_col
        self.output_stack = output_stack
        self.kernel_shape = (
            output_row * output_col * output_stack,
            self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * input_filter,
            self.filters)
        self.kernel = self.add_weight(shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(output_row, output_col, output_stack, self.filters),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=5, axes={channel_axis: input_filter})
        super(LocallyConnected3D, self).build(input_shape)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            stacks = input_shape[4]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            stacks = input_shape[3]

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding, self.strides[1])
        stacks = conv_utils.conv_output_length(stacks, self.kernel_size[2],
                                             self.padding, self.strides[2])

        if self.data_format == 'channels_first':
            return (input_shape[0], self.filters, rows, cols, stacks)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, stacks, self.filters)

    def call(self, inputs, **kwargs):
        output = K.local_conv(inputs, self.kernel, self.kernel_size, self.strides,
                              (self.output_row, self.output_col, self.output_stack),
                              self.data_format)

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format=self.data_format)

        output = self.activation(output)
        return output

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(LocallyConnected3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
