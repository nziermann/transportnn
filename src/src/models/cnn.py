import numpy as np
from src.layers import MassConversation3D, LandValueRemoval3D, LocallyConnected3D, WrapAroundPadding3D
from functools import partial
from src.visualization import save_data_for_visualization, save_as_netcdf
from src.data import get_training_data, get_volumes, get_landmask, load_netcdf_data, split_data, combine_data
import itertools
import json
import argparse
import subprocess
import os
import glob
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer, Conv3D, AveragePooling3D, UpSampling3D, BatchNormalization, Input, ZeroPadding3D
import tensorflow.keras.metrics
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class LocalNetwork(Model):

    def __init__(self, config, data, reduce_resolution=True):
        super(LocalNetwork, self).__init__()
        kernel_size = config.get('kernel_size', (5, 5, 5))
        activation = config.get('activation', 'relu')

        if config.get('land_removal_start', True):
            self.land_removal_start = LandValueRemoval3D(data['land'])

        padding_size = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        stride_size = (1, 1, 1)

        if reduce_resolution:
            # Special handling for (3, 3, 3) kernel size
            padding_size = (0, 1, 2)

        if reduce_resolution:
            stride_size = kernel_size

        self.zero = ZeroPadding3D(padding_size)
        self.local = LocallyConnected3D(1, kernel_size, activation=activation, strides=stride_size)

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

        self.add = Add()

        if config.get('land_removal', True):
            self.land_removal = LandValueRemoval3D(data['land'])

        if config.get('mass_normalization', True):
            self.mass_conversation = MassConversation3D(data['volumes'])

    def call(self, inputs, training=None, mask=None):
        x = inputs

        if self.land_removal_start:
            x = self.land_removal_start(x)

        for layer in self.sub_layers:
            x = layer(x)

        x = self.padded_conv(x, training=training)

        x = self.add([inputs, x])

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
        filters_2 = filters//2
        kernel_size = config.get('kernel_size', (5, 5, 5))
        activation = config.get('activation', 'relu')
        activation_last = config.get('activation_last', activation)
        batch_norm = config.get('batch_norm', False)
        pooling_type = config.get('pooling_type', AveragePooling3D)

        if config.get('land_removal_start', True):
            self.land_removal_start = LandValueRemoval3D(data['land'])

        if batch_norm:
            self.batch_norm_1 = BatchNormalization()
        self.conv_1 = Conv3D(filters, kernel_size, input_shape=input_shape, activation=activation, padding='same')
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

    return get_convolutional_autoencoder(data, config)


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


def cnn(data, x_train, y_train, x_val, y_val, params):
    print("Getting model with:")
    print(params)

    model = get_model(data, params)

    # early_stopping_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=10, patience=5,
    #                                                        restore_best_weights=True)
    # callbacks = [early_stopping_callback]
    callbacks = []

    optimizer = params.get('optimizer', 'adam')
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse', tensorflow.keras.metrics.mape, tensorflow.keras.metrics.mae])
    out = model.fit(x_train, y_train, epochs=params['epochs'], callbacks=callbacks, validation_data=(x_val, y_val))

    return out, model


def train_split_model(config, parameters):
    parameters['mass_normalization'] = [False]
    parameters['land_removal'] = [False]
    parameters['land_removal_start'] = [False]
    parameters['input_shape'] = [(15, 32, 64, 1)]

    # Temporarily use models without overlap for data
    def split(data, overlap=False):
        split_data_array = split_data(data, 2, 2, overlap=overlap)
        multi_split_data = []
        for single_split_data in split_data_array:
            multi_split_data.extend(split_data(single_split_data, 3, 2, overlap=overlap))

        return multi_split_data

    print("Loading data")
    x_train, y_train = get_training_data(config['data_dir'], config['samples'])
    assert not np.any(np.isnan(x)), "x_train contains nan data"
    assert not np.any(np.isnan(y)), "y_train contains nan data"
    x_val, y_val = get_training_data(config['validation_data'], config['samples'])
    assert not np.any(np.isnan(x)), "x_val contains nan data"
    assert not np.any(np.isnan(y)), "y_val contains nan data"
    print("Loaded data")

    x_train_split = split(x_train)
    y_train_split = split(y_train)
    x_val_split = split(x_val)
    y_val_split = split(y_val)

    predict_data = None
    if config['predict_data'] is not None:
        print("Loading validation data")
        predict_data = load_netcdf_data(config['predict_data'])
        predict_data = np.reshape(predict_data, (-1, 15, 64, 128, 1))
        assert not np.any(np.isnan(predict_data)), "Validation data contains nan data"
        print("Loaded validation data")
        predict_split = split(predict_data, False)

    # Temporarily disable special handling of land and mass
    data = {
    }

    print("Starting model")
    cnn_partial = partial(cnn, data)

    parameter_combinations = product_dict(**parameters)
    best_models = None
    first_models = None
    lowest_loss = np.inf
    for parameter_combination in parameter_combinations:
        models = []
        sum_loss = 0
        for i in range(len(x_train_split)):
            out, model = cnn_partial(x_train_split[i], y_train_split[i], x_val_split[i], y_val_split[i], parameter_combination)
            models.append(model)
            sum_loss = sum_loss + out.history['loss'][-1]

        if first_models is None:
            first_models = models

        if sum_loss < lowest_loss:
            best_models, lowest_loss = models, sum_loss

    if best_models is None:
        best_models = first_models

    if predict_data is not None:
        predict_validations_split(best_models, predict_data, predict_split, config)

    for i in range(len(best_models)):
        model = best_models[i]
        model.save(f'{config["job_dir"]}/best_model_{i}.h5')


def predict_validations_split(models, validation_data, validation_data_split, config):
    starts = map(lambda validation_element: validation_element[0], validation_data_split)
    starts = list(map(lambda start: start[np.newaxis], starts))
    samples = np.size(validation_data_split[0], 0)
    predictions = np.full((samples-1, 15, 64, 128, 1), np.nan)
    for i in range(len(validation_data_split)):
        sub_predictions = []
        for j, model in enumerate(models, start=0):
            prediction = model.predict(starts[j])
            assert not np.any(np.isnan(prediction)), "Model predicted nan"
            sub_predictions.append(prediction)

        predictions[i] = combine_data(sub_predictions)
        assert not np.any(np.isnan(predictions[i])), "Predictions contain nan after combining data"

        starts = sub_predictions

    save_as_netcdf(config['grid_file'], f'{config["job_dir"]}/model_predictions_validation.nc',
                   predictions, validation_data[1:])


def train_models(config, parameters):
    print("Loading data")
    x_train, y_train = get_training_data(config['data_dir'], config['samples'])
    assert not np.any(np.isnan(x_train)), "x_train contains nan data"
    assert not np.any(np.isnan(y_train)), "y_train contains nan data"
    x_val, y_val = get_training_data(config['validation_data'], np.inf)
    assert not np.any(np.isnan(x_val)), "x_val contains nan data"
    assert not np.any(np.isnan(y_val)), "y_val contains nan data"

    print("Loaded data")

    predict_data = None
    if config['predict_data'] is not None:
        print("Loading validation data")
        predict_data = load_netcdf_data(config['predict_data'])
        predict_data = np.reshape(predict_data, (-1, 15, 64, 128, 1))
        assert not np.any(np.isnan(predict_data)), "Validation data contains nan data"
        print("Loaded validation data")

    print("Loading volumes")
    volumes = np.reshape(get_volumes(config['volumes_file']), (1, 15, 64, 128, 1))
    assert not np.any(np.isnan(volumes)), "Volumes data contains nan data"
    print("Loaded volumes")

    print("Loading land")
    land = np.reshape(get_landmask(config['grid_file']), (1, 15, 64, 128, 1))
    assert not np.any(np.isnan(land)), "Land data contains nan data"
    print("Loaded land")

    data = {
        'volumes': volumes,
        'land': land
    }

    print("Starting model")
    cnn_partial = partial(cnn, data)

    parameter_combinations = product_dict(**parameters)
    best_model = None
    first_model = None
    lowest_loss = np.inf
    for parameter_combination in parameter_combinations:
        out, model = cnn_partial(x_train, y_train, x_val, y_val, parameter_combination)

        if first_model is None:
            first_model = model

        model_loss = out.history['loss'][-1]
        print(f'Model loss: {model_loss}')

        if model_loss < lowest_loss:
            best_model, lowest_loss = model, model_loss

    if best_model is None:
        best_model = first_model

    if predict_data is not None:
        predict_validations(best_model, predict_data, config)

    best_model.save(f'{config["job_dir"]}/best_model', save_format='tf')
    save_data_for_visualization(best_model, config['data_dir'], config['samples'], config['grid_file'],
                                config['job_dir'])


def predict_validations(model, validation_data, config):
    start = validation_data[0]
    start = start[np.newaxis]
    samples = np.size(validation_data, 0)
    predictions = np.full((samples - 1, 15, 64, 128, 1), np.nan)
    for i, y in enumerate(validation_data[1:], start=1):
        start = model.predict(start)
        predictions[i - 1] = start

    assert not np.any(np.isnan(predictions)), "Predictions contains nan data"

    save_as_netcdf(config['grid_file'], f'{config["job_dir"]}/model_predictions_validation.nc',
                   predictions, validation_data[1:])


def get_model_summaries(config, parameters):
    data = get_dummy_data()
    parameters_product = product_dict(**parameters)

    for parameters_selection in parameters_product:
        model = get_model(data, parameters_selection)
        print(parameters_selection)
        model.summary()


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def get_dummy_data():
    volumes = np.zeros((1, 15, 64, 128, 1))
    land = np.zeros((1, 15, 64, 128, 1))

    data = {
        'volumes': volumes,
        'land': land
    }

    return data


def main():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    parameters = {
        'filter_exponent': [2, 4, 6],
        'depth': [2, 4, 6],
        'kernel_size': [3],
        'activation': ['relu'],
        #'epochs': [100],
        'epochs': [5],
        'batch_norm': [False],
        'optimizer': ['adam'],
        'mass_normalization': [True, False],
        'land_removal': [True, False],
        'land_removal_start': [True, False],
        'model_type': ['simple', 'climatenn']
    }

    defaults = {
        'data_dir': "/storage/data/3d/smooth",
        'validation_data': "/storage/data/3d/validation",
        'volumes_file': '/storage/other/normalizedVolumes.nc',
        'grid_file': '/storage/other/mitgcm-128x64-grid-file.nc',
        'job_dir': "/artifacts",
        # 'samples': np.inf
        'samples': 220
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters_file", help="")
    parser.add_argument("--data-dir", help="", default=defaults['data_dir'])
    parser.add_argument("--validation-data", default=defaults['validation_data'])
    parser.add_argument("--predict-data", default=None)
    parser.add_argument("--job-dir", help="", default=defaults['job_dir'])
    parser.add_argument("--volumes-file", help="", default=defaults['volumes_file'])
    parser.add_argument("--grid-file", help="", default=defaults['grid_file'])
    parser.add_argument("--samples", help="", default=defaults['samples'])
    parser.add_argument("--split", help="", action="store_true", default=False)
    parser.add_argument("--print-summaries", help="", action="store_true", default=False)
    parser.add_argument("--download-from", help="", default=None)
    parser.add_argument("--upload-to", help="", default=None)
    args = parser.parse_args()

    if not args.parameters_file is None:
        with open(args.parameters_file, "r") as parameters_file:
            parameters = json.load(parameters_file)

    config = {
        'data_dir': args.data_dir,
        'job_dir': args.job_dir,
        'volumes_file': args.volumes_file,
        'grid_file': args.grid_file,
        'samples': args.samples,
        'validation_data': args.validation_data,
        'predict_data': args.predict_data
    }

    if args.download_from is not None:
        subprocess.check_call(['gsutil', '-m', 'cp', '-r', args.download_from, '/tmp'])

    print(config)
    print(os.environ.get('HDF5_USE_FILE_LOCKING'))

    if args.print_summaries:
        get_model_summaries(config, parameters)
    elif args.split:
        train_split_model(config, parameters)
    else:
        train_models(config, parameters)

    print("Upload to:")
    print(args.upload_to)
    if args.upload_to is not None:
        print("Uploading")
        print(f'Upload to: {args.upload_to}')
        print(f'Job dir: {config["job_dir"]}')
        print(f'Call: gsutil -m cp -r {config["job_dir"]} {args.upload_to}')

        print("Files in folder:")
        files = [f for f in glob.glob(config['job_dir'] + "**/*.nc", recursive=True)]

        for f in files:
            print(f)

        subprocess.check_call(['gsutil', '-m', 'cp', '-r', config['job_dir'], args.upload_to])


if __name__ == "__main__":
    main()
