from src.layers import MassConversation3D, LandValueRemoval3D, WrapAroundPadding3D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ZeroPadding1D, Cropping3D, Permute, Reshape, LocallyConnected1D, TimeDistributed, Concatenate, ZeroPadding2D, Layer, Conv3D, AveragePooling3D, UpSampling3D, BatchNormalization, ZeroPadding3D, Activation, Add, Cropping3D
import tensorflow as tf



class LocalNetwork(Model):

    def __init__(self, config, data, reduce_resolution=False, residual=False):
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

        self.downsampling = AveragePooling3D(pool_size=(3, 4, 4))

        depth_padding = (1, 0, 0)
        depth_permutation = (3, 2, 1, 4)
        self.zero_depth = ZeroPadding3D(depth_padding)
        self.permutation_depth_start = Permute(depth_permutation)
        self.reshape_depth_start = Reshape(((15//3+2)*64//4*128//4, 1))

        self.zero_depth_1d = ZeroPadding1D(1)
        self.local_depth = LocallyConnected1D(1, 3, activation=activation)

        self.reshape_depth_end = Reshape((128//4, 64//4, 15//3+2, 1))
        self.permutation_depth_end = Permute(depth_permutation)

        self.cropping_depth = Cropping3D(depth_padding)
        
        longitude_padding = (0, 0, 1)
        longitude_permutation = (1, 2, 3, 4)
        self.zero_longitude = ZeroPadding3D(longitude_padding)
        self.permutation_longitude_start = Permute(longitude_permutation)
        self.reshape_longitude_start = Reshape((15//3*64//4*(128//4+2), 1))

        self.zero_longitude_1d = ZeroPadding1D(1)
        self.local_longitude = LocallyConnected1D(1, 3, activation=activation)

        self.reshape_longitude_end = Reshape((15//3, 64//4, 128//4+2, 1))
        self.permutation_longitude_end = Permute(longitude_permutation)

        self.cropping_longitude = Cropping3D(longitude_padding)
        
        latitude_padding = (0, 1, 0)
        latitude_permutation = (1, 3, 2, 4)
        self.zero_latitude = ZeroPadding3D(latitude_padding)
        self.permutation_latitude_start = Permute(latitude_permutation)
        self.reshape_latitude_start = Reshape((15//3*(64//4+2)*128//4, 1))

        self.zero_latitude_1d = ZeroPadding1D(1)
        self.local_latitude = LocallyConnected1D(1, 3, activation=activation)

        self.reshape_latitude_end = Reshape((15//3, 128//4, 64//4+2, 1))
        self.permutation_latitude_end = Permute(latitude_permutation)

        self.cropping_latitude = Cropping3D(latitude_padding)

        self.upsampling = UpSampling3D(size=(3, 4, 4))

        if reduce_resolution:
            self.upsampling = UpSampling3D(stride_size)
            self.cropping = Cropping3D(padding_size)

        if residual:
            self.residual = Add()

        if config.get('land_removal', True):
            self.land_removal = LandValueRemoval3D(data['land'])

        if config.get('mass_normalization', True):
            self.mass_normalization = MassConversation3D(data['volumes'])

    def call(self, inputs, training=None, mask=None):
        x = inputs
        if hasattr(self, 'land_removal_start'):
            x = self.land_removal_start(x)

        x = self.downsampling(x)

        x = self.zero_depth(x)
        x = self.permutation_depth_start(x)
        x = self.reshape_depth_start(x)

        x = self.zero_depth_1d(x)
        x = self.local_depth(x)

        x = self.reshape_depth_end(x)
        x = self.permutation_depth_end(x)

        x = self.cropping_depth(x)

        x = self.zero_longitude(x)
        x = self.permutation_longitude_start(x)
        x = self.reshape_longitude_start(x)

        x = self.zero_longitude_1d(x)
        x = self.local_longitude(x)

        x = self.reshape_longitude_end(x)
        x = self.permutation_longitude_end(x)

        x = self.cropping_longitude(x)

        x = self.zero_latitude(x)
        x = self.permutation_latitude_start(x)
        x = self.reshape_latitude_start(x)

        x = self.zero_latitude_1d(x)
        x = self.local_latitude(x)

        x = self.reshape_latitude_end(x)
        x = self.permutation_latitude_end(x)

        x = self.cropping_latitude(x)

        x = self.upsampling(x)

        #if hasattr(self, 'upsampling'):
        #    x = self.upsampling(x)

        if hasattr(self, 'cropping'):
            x = self.cropping(x)

        if hasattr(self, 'residual'):
            x = self.residual([x, inputs])

        if hasattr(self, 'land_removal'):
            x = self.land_removal(x)

        if hasattr(self, 'mass_normalization'):
            x = self.mass_normalization([inputs, x])

        return x


class PaddedConv3D(Layer):
    def __init__(self, filters, kernel_size, activation, batch_norm):
        super(PaddedConv3D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_norm = batch_norm

        if batch_norm:
            self.batch_norm_layer = BatchNormalization()

        self.zero_padding = ZeroPadding3D((1, 0, 0))
        self.wrap_around_padding = WrapAroundPadding3D((0, 1, 1))
        self.conv = Conv3D(filters, kernel_size, activation=activation)

    def call(self, inputs, training=None):
        x = inputs

        if hasattr(self, 'batch_norm_layer'):
            x = self.batch_norm_layer(x, training=training)

        x = self.zero_padding(x)
        x = self.wrap_around_padding(x)
        x = self.conv(x)

        return x

    def get_config(self):
        config = super(PaddedConv3D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'batch_norm': self.batch_norm
        })
        return config


class SimpleConvolutionAutoencoder(Model):

    def __init__(self, config, data):
        super(SimpleConvolutionAutoencoder, self).__init__()
        self.config = config
        self.data = data
        self.residual = config.get('residual', False)

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
        self.sub_activations = []
        for i in range(depth):
            padded_conv = PaddedConv3D(filters, kernel_size, None, batch_norm)
            activation = Activation(activation)
            self.sub_layers.append(padded_conv)
            self.sub_activations.append(activation)

        self.padded_conv = PaddedConv3D(1, kernel_size, activation_last, batch_norm)

        if self.residual:
            self.add = Add()

        if config.get('land_removal', True):
            self.land_removal = LandValueRemoval3D(data['land'])

        if config.get('mass_normalization', True):
            self.mass_conversation = MassConversation3D(data['volumes'])

    def call(self, inputs, training=None, mask=None):
        x = inputs

        if hasattr(self, 'land_removal_start'):
            x = self.land_removal_start(x)

        if self.residual:
            shortcut = x

        for i, layer in enumerate(self.sub_layers, start=1):
            x = layer(x)
            current_residual = self.residual and i%2

            if current_residual:
                x = self.add([x, shortcut])

            activation = self.sub_activations.__getitem__(i-1)
            x = activation(x)

            if current_residual:
                shortcut = x

        x = self.padded_conv(x, training=training)

        if hasattr(self, 'land_removal'):
            x = self.land_removal(x)

        if hasattr(self, 'mass_conversation'):
            x = self.mass_conversation([inputs, x])

        return x

    def get_config(self):
        config = super(SimpleConvolutionAutoencoder, self).get_config()
        config.update({
            'config': self.config,
            'data': self.data,
            'residual': self.residual
        })

        return config


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
        self.pw_1 = WrapAroundPadding3D((0, 1, 1))
        self.pz_1 = ZeroPadding3D((1, 0, 0))
        self.conv_1 = Conv3D(filters, kernel_size, activation=activation)
        self.pooling_1 = pooling_type((1, 2, 2))

        if batch_norm:
            self.batch_norm_2 = BatchNormalization()
        self.pw_2 = WrapAroundPadding3D((0, 1, 1))
        self.pz_2 = ZeroPadding3D((1, 0, 0))
        self.conv_2 = Conv3D(filters_2, kernel_size, activation=activation)
        self.pooling_2 = pooling_type((1, 2, 2))

        if batch_norm:
            self.batch_norm_3 = BatchNormalization()
        self.pw_3 = WrapAroundPadding3D((0, 1, 1))
        self.pz_3 = ZeroPadding3D((1, 0, 0))
        self.conv_3 = Conv3D(filters_2, kernel_size, activation=activation)
        self.pooling_3 = pooling_type((3, 2, 2))

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
        if batch_norm:
            self.batch_norm_4 = BatchNormalization()
        self.pw_4 = WrapAroundPadding3D((0, 1, 1))
        self.pz_4 = ZeroPadding3D((1, 0, 0))
        self.conv_4 = Conv3D(filters_2, kernel_size, activation=activation)
        self.up_4 = UpSampling3D((3, 2, 2))

        if batch_norm:
            self.batch_norm_5 = BatchNormalization()
        self.pw_5 = WrapAroundPadding3D((0, 1, 1))
        self.pz_5 = ZeroPadding3D((1, 0, 0))
        self.conv_5 = Conv3D(filters_2, kernel_size, activation=activation)
        self.up_5 = UpSampling3D((1, 2, 2))

        if batch_norm:
            self.batch_norm_6 = BatchNormalization()
        self.pw_6 = WrapAroundPadding3D((0, 1, 1))
        self.pz_6 = ZeroPadding3D((1, 0, 0))
        self.conv_6 = Conv3D(filters, kernel_size, activation=activation)
        self.up_6 = UpSampling3D((1, 2, 2))

        # cnn = sub_model(input)
        # output = Conv3D(1, kernel_size, activation=activation_last, padding='same')(cnn)
        self.output_conv = Conv3D(1, kernel_size, activation=activation_last)

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
        x = self.pw_1(x)
        x = self.pz_1(x)
        x = self.conv_1(x)
        x = self.pooling_1(x)

        if hasattr(self, 'batch_norm_2'):
            x = self.batch_norm_2(x, training=training)
        x = self.pw_2(x)
        x = self.pz_2(x)
        x = self.conv_2(x)
        x = self.pooling_2(x)

        if hasattr(self, 'batch_norm_3'):
            x = self.batch_norm_3(x, training=training)
        x = self.pw_3(x)
        x = self.pz_3(x)
        x = self.conv_3(x)
        x = self.pooling_3(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
        if hasattr(self, 'batch_norm_4'):
            x = self.batch_norm_4(x, training=training)
        x = self.pw_4(x)
        x = self.pz_4(x)
        x = self.conv_4(x)
        x = self.up_4(x)

        if hasattr(self, 'batch_norm_5'):
            x = self.batch_norm_5(x, training=training)
        x = self.pw_5(x)
        x = self.pz_5(x)
        x = self.conv_5(x)
        x = self.up_5(x)

        if hasattr(self, 'batch_norm_6'):
            x = self.batch_norm_6(x, training=training)
        x = self.pw_6(x)
        x = self.pz_6(x)
        x = self.conv_6(x)
        x = self.up_6(x)

        x = self.output_conv(x)

        if hasattr(self, 'land_removal'):
            x = self.land_removal(x)

        if hasattr(self, 'mass_normalization'):
            x = self.mass_normalization([inputs, x])

        return x

    def get_config(self):
        config = super(ConvolutionalAutoencoder, self).get_config()
        config.update({
            'config': self.config,
            'data': self.data
        })
        return config


def get_model(data, config):
    model_type = config['model_type']

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
    model = ConvolutionalAutoencoder(config, data)

    return model
