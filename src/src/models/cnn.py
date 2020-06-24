import numpy as np
from src.layers import MassConversation3D, LandValueRemoval3D, LocallyConnected3D, WrapAroundPadding3D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed, LocallyConnected2D, ZeroPadding2D, Layer, Conv3D, AveragePooling3D, UpSampling3D, BatchNormalization, ZeroPadding3D, Activation, Add, Cropping3D
import tensorflow as tf



class LocalNetwork(Model):

    def __init__(self, config, data, reduce_resolution=False):
        super(LocalNetwork, self).__init__()
        kernel_size = config.get('kernel_size', (5, 5, 5))
        activation = config.get('activation', 'relu')

        if config.get('land_removal_start', True):
            self.land_removal_start = LandValueRemoval3D(data['land'])

        if type(kernel_size) is int:
            kernel_size = [kernel_size, kernel_size]

        padding_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        stride_size = (1, 1)

        if reduce_resolution:
            # Special handling for (3, 3, 3) kernel size
            padding_size = (0, 1, 2)

        if reduce_resolution:
            stride_size = kernel_size

        def mass_transport_regularizer(x):
            return tf.abs(tf.reduce_sum(x))

        self.zero = TimeDistributed(ZeroPadding2D(padding_size))

        # self.locals_1 = []

        # def create_local():
        #     return LocallyConnected2D(1, kernel_size, activation=activation, strides=stride_size,
        #                                                 # kernel_regularizer=mass_transport_regularizer,
        #                                                 use_bias=False,
        #                                                 kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001))

        # for i in range(0, 1):
        #     local = create_local()
        #     for j in range(0, 15):
        #         self.locals_1.append(local)

        # self.local = create_local():
        self.local = TimeDistributed(LocallyConnected2D(1, kernel_size, activation=activation, strides=stride_size,
                                                  # kernel_regularizer=mass_transport_regularizer,
                                                  use_bias=True))

        if reduce_resolution:
            self.upsampling = UpSampling3D(stride_size)
            self.cropping = Cropping3D(padding_size)

        if config.get('land_removal', True):
            self.land_removal = LandValueRemoval3D(data['land'])

        if config.get('mass_normalization', True):
            self.mass_normalization = MassConversation3D(data['volumes'])

    def call(self, inputs):
        x = inputs
        if hasattr(self, 'land_removal_start'):
            x = self.land_removal_start(x)

        x = self.zero(x)

        # output_list = []
        # for i, local in enumerate(self.locals_1):
        #     output_list.append(local(x[:, i, :, :, :]))

        # x = tf.stack(output_list, 1)

        x = self.local(x)

        if hasattr(self, 'upsampling'):
            x = self.upsampling(x)

        if hasattr(self, 'cropping'):
            x = self.cropping(x)

        if hasattr(self, 'land_removal'):
            x = self.land_removal(x)

        if hasattr(self, 'mass_normalization'):
            x = self.mass_normalization([inputs, x])

        return x


class PaddedConv3D(Layer):
    def __init__(self, filters, kernel_size, activation, batch_norm, residual):
        super(PaddedConv3D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_norm = batch_norm
        self.residual = residual

        if batch_norm:
            self.batch_norm_layer = BatchNormalization()

        self.zero_padding = ZeroPadding3D((1, 0, 0))
        self.wrap_around_padding = WrapAroundPadding3D((0, 1, 1))
        self.conv = Conv3D(filters, kernel_size, activation=None)

        if residual:
            self.add = Add()

        self.activation = Activation(activation)

    def call(self, inputs, training=None):
        x = inputs

        if hasattr(self, 'batch_norm_layer'):
            x = self.batch_norm_layer(x, training=training)

        x = self.zero_padding(x)
        x = self.wrap_around_padding(x)
        x = self.conv(x)

        if hasattr(self, 'add'):
            x = self.add([x, inputs])

        x = self.activation(x)

        return x

    def get_config(self):
        return {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'batch_norm': self.batch_norm,
            'residual': self.residual
        }


class SimpleConvolutionAutoencoder(Model):

    def __init__(self, config, data):
        super(SimpleConvolutionAutoencoder, self).__init__()
        self.config = config
        self.data = data

        filter_exponent = config.get('filter_exponent', 4)
        filters = int(2 ** filter_exponent)
        kernel_size = config.get('kernel_size', (5, 5, 5))
        activation = config.get('activation', 'relu')
        activation_last = config.get('activation_last', activation)
        batch_norm = config.get('batch_norm', False)
        depth = config.get('depth', 6)

        if config.get('land_removal_start', True):
            self.land_removal_start = LandValueRemoval3D(data['land'])

        self.sub_layers = []
        for i in range(depth):
            padded_conv = PaddedConv3D(filters, kernel_size, activation, batch_norm, False)
            self.sub_layers.append(padded_conv)

        self.padded_conv = PaddedConv3D(1, kernel_size, activation_last, batch_norm, False)

        # Todo: Reactivate if convolutions are wanted
        # self.add = Add()

        if config.get('land_removal', True):
            self.land_removal = LandValueRemoval3D(data['land'])

        if config.get('mass_normalization', True):
            self.mass_conversation = MassConversation3D(data['volumes'])

    def call(self, inputs, training=None, mask=None):
        x = inputs

        if hasattr(self, 'land_removal_start'):
            x = self.land_removal_start(x)

        for layer in self.sub_layers:
            x = layer(x)

        x = self.padded_conv(x, training=training)

        # Todo: Reactivate if convolutions are wanted
        # x = self.add([inputs, x])

        if hasattr(self, 'land_removal'):
            x = self.land_removal(x)

        if hasattr(self, 'mass_conversation'):
            x = self.mass_conversation([inputs, x])

        return x

    def get_config(self):
        return {
            'config': self.config,
            'data': self.data
        }


class NoneModel(Model):

    def __init__(self):
        super(NoneModel, self).__init__()

    def call(self, inputs, training=None, mask=None):
        return inputs


class ConvolutionalAutoencoder(Model):

    def __init__(self, config, data):
        super(ConvolutionalAutoencoder, self).__init__()
        self.config = config
        self.data = data

        filter_exponent = config.get('filter_exponent', 4)
        filters = int(2 ** filter_exponent)
        filters_2 = filters // 2
        kernel_size = config.get('kernel_size', (5, 5, 5))
        activation = config.get('activation', 'relu')
        activation_last = config.get('activation_last', activation)
        batch_norm = config.get('batch_norm', False)
        pooling_type = config.get('pooling_type', AveragePooling3D)

        if config.get('land_removal_start', True):
            self.land_removal_start = LandValueRemoval3D(data['land'])

        if batch_norm:
            self.batch_norm_1 = BatchNormalization()
        self.conv_1 = Conv3D(filters, kernel_size, activation=activation, padding='same')
        self.pooling_1 = pooling_type((1, 2, 2), padding='same')

        if batch_norm:
            self.batch_norm_2 = BatchNormalization()
        self.conv_2 = Conv3D(filters_2, kernel_size, activation=activation, padding='same')
        self.pooling_2 = pooling_type((1, 2, 2), padding='same')

        if batch_norm:
            self.batch_norm_3 = BatchNormalization()
        self.conv_3 = Conv3D(filters_2, kernel_size, activation=activation, padding='same')
        self.pooling_3 = pooling_type((3, 2, 2), padding='same')

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
        if batch_norm:
            self.batch_norm_4 = BatchNormalization()
        self.conv_4 = Conv3D(filters_2, kernel_size, activation=activation, padding='same')
        self.up_4 = UpSampling3D((3, 2, 2))

        if batch_norm:
            self.batch_norm_5 = BatchNormalization()
        self.conv_5 = Conv3D(filters_2, kernel_size, activation=activation, padding='same')
        self.up_5 = UpSampling3D((1, 2, 2))

        if batch_norm:
            self.batch_norm_6 = BatchNormalization()
        self.conv_6 = Conv3D(filters, kernel_size, activation=activation, padding='same')
        self.up_6 = UpSampling3D((1, 2, 2))

        # cnn = sub_model(input)
        # output = Conv3D(1, kernel_size, activation=activation_last, padding='same')(cnn)
        self.output_conv = Conv3D(1, kernel_size, activation=activation_last, padding='same')

        if config.get('land_removal', True):
            self.land_removal = LandValueRemoval3D(data['land'])

        if config.get('mass_normalization', True):
            self.mass_normalization = MassConversation3D(data['volumes'])

    def call(self, inputs, training=None, mask=None):
        x = inputs

        if hasattr(self, 'land_removal_start'):
            x = self.land_removal_start(x)

        if hasattr(self, 'batch_norm_1'):
            x = self.batch_norm_1(x, training=training)
        x = self.conv_1(x)
        x = self.pooling_1(x)

        if hasattr(self, 'batch_norm_2'):
            x = self.batch_norm_2(x, training=training)
        x = self.conv_2(x)
        x = self.pooling_2(x)

        if hasattr(self, 'batch_norm_3'):
            x = self.batch_norm_3(x, training=training)
        x = self.conv_3(x)
        x = self.pooling_3(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
        if hasattr(self, 'batch_norm_4'):
            x = self.batch_norm_4(x, training=training)
        x = self.conv_4(x)
        x = self.up_4(x)

        if hasattr(self, 'batch_norm_5'):
            x = self.batch_norm_5(x, training=training)
        x = self.conv_5(x)
        x = self.up_5(x)

        if hasattr(self, 'batch_norm_6'):
            x = self.batch_norm_6(x, training=training)
        x = self.conv_6(x)
        x = self.up_6(x)

        x = self.output_conv(x)

        if hasattr(self, 'land_removal'):
            x = self.land_removal(x)

        if hasattr(self, 'mass_normalization'):
            x = self.mass_normalization(x)

        return x

    def get_config(self):
        return {
            'config': self.config,
            'data': self.data
        }


def get_model(data, config):
    model_type = config.get('model_type', 'climatenn')

    if model_type == 'simple':
        return get_simple_convolutional_autoencoder(data, config)

    if model_type == 'none':
        return get_none_model(data, config)

    if model_type == 'local':
        return get_local_network(data, config)

    if model_type == 'climatenn':
        return get_convolutional_autoencoder(data, config)

    print("Unknown model type wanted")
    exit()


def get_none_model(data, config):
    model = NoneModel()

    return model


def get_local_network(data, config):
    model = LocalNetwork(config, data)

    return model


def get_simple_convolutional_autoencoder(data, config):
    model = SimpleConvolutionAutoencoder(config, data)

    return model


# Currently allowed parameters of config are
# filter_exponent
# kernel_size
# pooling_type
# activation
# activation_last
def get_convolutional_autoencoder(data, config):
    model = SimpleConvolutionAutoencoder(config, data)

    return model
