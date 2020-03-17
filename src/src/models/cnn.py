from keras.models import Sequential, Model
from keras.layers import Conv3D, AveragePooling3D, UpSampling3D, BatchNormalization, Input, ZeroPadding3D
import keras.metrics
import numpy as np
from src.layers import MassConversation3D, LandValueRemoval3D, LocallyConnected3D
from functools import partial
from src.visualization import save_data_for_visualization, save_as_netcdf
from src.data import get_training_data, get_volumes, get_landmask, load_netcdf_data
import itertools
import json
import argparse
import subprocess
import os
import keras.backend as K
import glob

def get_model(data, config):
    model_type = config.get('model_type', 'climatenn')

    if model_type == 'simple':
        return get_simple_convolutional_autoencoder(data, config)

    if model_type == 'local':
        return get_local_network(data, config)

    return get_convolutional_autoencoder(data, config)

def get_local_network(data, config):
    filter_exponent = config.get('filter_exponent', 4)
    filters = int(2 ** filter_exponent)
    kernel_size = config.get('kernel_size', (5, 5, 5))
    activation = config.get('activation', 'relu')
    activation_last = config.get('activation_last', activation)
    batch_norm = config.get('batch_norm', False)
    depth = config.get('depth', 6)

    input_shape = (15, 64, 128, 1)

    input_layer = Input(shape=input_shape)
    start_layer = input_layer

    if config.get('land_removal_start', True):
        start_layer = LandValueRemoval3D(data['land'])(start_layer)

    padding_size = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
    print(f'Kernel size {kernel_size}')
    print(f'Padding size: {padding_size}')

    sub_model = Sequential()

    zero = ZeroPadding3D(padding_size, input_shape=input_shape)
    sub_model.add(zero)
    local = LocallyConnected3D(1, kernel_size, activation=activation)
    sub_model.add(local)
    output = sub_model(start_layer)

    if config.get('land_removal', True):
        output = LandValueRemoval3D(data['land'])(output)

    if config.get('mass_normalization', True):
        output = MassConversation3D(data['volumes'])([start_layer, output])

    model = Model(inputs=input_layer, outputs=output)

    return model


def get_simple_convolutional_autoencoder(data, config):
    filter_exponent = config.get('filter_exponent', 4)
    filters = int(2 ** filter_exponent)
    kernel_size = config.get('kernel_size', (5, 5, 5))
    activation = config.get('activation', 'relu')
    activation_last = config.get('activation_last', activation)
    batch_norm = config.get('batch_norm', False)
    depth = config.get('depth', 6)

    input_shape = (15, 64, 128, 1)

    input_layer = Input(shape=input_shape)
    start_layer = input_layer

    if config.get('land_removal_start', True):
        start_layer = LandValueRemoval3D(data['land'])(start_layer)

    sub_model = Sequential()

    for i in range(depth):
        if batch_norm:
            sub_model.add(BatchNormalization(input_shape=input_shape))
        convi = Conv3D(filters, kernel_size, input_shape=input_shape, activation=activation, padding='same')
        sub_model.add(convi)

    conv = Conv3D(1, kernel_size, activation=activation_last, padding='same')
    sub_model.add(conv)
    output = sub_model(start_layer)

    sub_model.summary()

    if config.get('land_removal', True):
        output = LandValueRemoval3D(data['land'])(output)

    if config.get('mass_normalization', True):
        output = MassConversation3D(data['volumes'])([start_layer, output])

    model = Model(inputs=input_layer, outputs=output)

    return model

def get_non_shared_convolutional_autoencoder(data, config):
    filter_exponent = config.get('filter_exponent', 4)
    filters = int(2 ** filter_exponent)
    kernel_size = config.get('kernel_size', (5, 5, 5))
    activation = config.get('activation', 'relu')
    activation_last = config.get('activation_last', activation)
    batch_norm = config.get('batch_norm', False)
    depth = config.get('depth', 6)

    input_shape = (15, 64, 128, 1)

    input_layer = Input(shape=input_shape)
    start_layer = input_layer

    if config.get('land_removal_start', True):
        start_layer = LandValueRemoval3D(data['land'])(start_layer)

    sub_model = Sequential()

    for i in range(depth):
        if batch_norm:
            sub_model.add(BatchNormalization(input_shape=input_shape))
        sub_model.add(Conv3D(filters, kernel_size, input_shape=input_shape, activation=activation, padding='same'))

    sub_model.add(Conv3D(1, kernel_size, activation=activation_last, padding='same'))
    output = sub_model(start_layer)

    if config.get('land_removal', True):
        output = LandValueRemoval3D(data['land'])(output)

    if config.get('mass_normalization', True):
        output = MassConversation3D(data['volumes'])([start_layer, output])

    model = Model(inputs=input_layer, outputs=output)

    return model

# Currently allowed parameters of config are
# filter_exponent
# kernel_size
# pooling_type
# activation
# activation_last
def get_convolutional_autoencoder(data, config):
    filter_exponent = config.get('filter_exponent', 4)
    filters = int(2**filter_exponent)
    filters_2 = int(filters/2)
    kernel_size = config.get('kernel_size', (5, 5, 5))
    pooling_type = config.get('pooling_type', AveragePooling3D)
    activation = config.get('activation', 'relu')
    activation_last = config.get('activation_last', activation)
    batch_norm = config.get('batch_norm', False)

    input_shape = (15, 64, 128, 1)

    input_layer = Input(shape=input_shape)
    start_layer = input_layer

    if config.get('land_removal_start', True):
        start_layer = LandValueRemoval3D(data['land'])(start_layer)

    sub_model = Sequential()
    if batch_norm:
        sub_model.add(BatchNormalization(input_shape=input_shape))
    sub_model.add(Conv3D(filters, kernel_size, input_shape=input_shape, activation=activation, padding='same'))
    sub_model.add(pooling_type((1, 2, 2), padding='same'))
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv3D(filters_2, kernel_size, activation=activation, padding='same'))
    sub_model.add(pooling_type((1, 2, 2), padding='same'))
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv3D(filters_2, kernel_size, activation=activation, padding='same'))
    sub_model.add(pooling_type((3, 2, 2), padding='same'))

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv3D(filters_2, kernel_size, activation=activation, padding='same'))
    sub_model.add(UpSampling3D((3, 2, 2)))
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv3D(filters_2, kernel_size, activation=activation, padding='same'))
    sub_model.add(UpSampling3D((1, 2, 2)))
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv3D(filters, kernel_size, activation=activation, padding='same'))
    sub_model.add(UpSampling3D((1, 2, 2)))
    if batch_norm:
        sub_model.add(BatchNormalization())

    # cnn = sub_model(input)
    # output = Conv3D(1, kernel_size, activation=activation_last, padding='same')(cnn)
    sub_model.add(Conv3D(1, kernel_size, activation=activation_last, padding='same'))
    output = sub_model(start_layer)

    if config.get('land_removal', True):
        output = LandValueRemoval3D(data['land'])(output)


    if config.get('mass_normalization', True):
        output = MassConversation3D(data['volumes'])([start_layer, output])

    model = Model(inputs=input_layer, outputs=output)

    return model

def cnn(data, x_train, y_train, x_val, y_val, params):
    print("Getting model with:")
    print(params)

    model = get_model(data, params)

    # early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=10, patience=5,
    #                                                        restore_best_weights=True)
    # callbacks = [early_stopping_callback]
    callbacks = []

    optimizer = params.get('optimizer', 'adam')
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse', keras.metrics.mape, keras.metrics.mae])
    out = model.fit(x_train, y_train, epochs=params['epochs'], callbacks=callbacks, validation_data=(x_val, y_val))

    return out, model


def train_models(config, parameters):
    print("Loading data")
    x, y = get_training_data(config['data_dir'], config['samples'])
    assert not np.any(np.isnan(x)), "X contains nan data"
    assert not np.any(np.isnan(y)), "Y contains nan data"
    print("Loaded data")

    validation_data = None
    if config['validation_data'] is not None:
        print("Loading validation data")
        validation_data = load_netcdf_data(config['validation_data'])
        assert not np.any(np.isnan(validation_data)), "Validation data contains nan data"
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
        x_train, y_train = x, y
        x_val, y_val = x, y
        out, model = cnn_partial(x_train, y_train, x_val, y_val, parameter_combination)

        if first_model is None:
            first_model = model

        model_loss = out.history['loss'][-1]
        print(f'Model loss: {model_loss}')

        if model_loss < lowest_loss:
            best_model, lowest_loss = model, model_loss

    if best_model is None:
        best_model = first_model

    if validation_data is not None:
        predict_validations(best_model, validation_data, config)

    save_data_for_visualization(best_model, config['data_dir'], config['samples'], config['grid_file'], config['job_dir'])

def predict_validations(model, validation_data, config):
    start = validation_data[0]
    samples = np.size(validation_data, 0)
    predictions = np.full((samples, 15, 64, 128, 1), np.nan)
    for i, y in enumerate(validation_data[1:], start=1):
        predictions[i] = model.predict(start)
        start = predictions[i]

    save_as_netcdf(config['grid_file'], f'{config["job_dir"]}/model_predictions_validation.nc', predictions, validation_data)

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

p = {
    'filter_exponent': [4],
    'kernel_size': [(3, 3, 3)],
    'activation': ['elu'],
    #'epochs': [100],
    'epochs': [1],
    'batch_norm': [False],
    'optimizer': ['adam'],
    #'mass_normalization': [True, False],
    'mass_normalization': [False],
    'land_removal': [True],
    'land_removal_start': [True],
    'model_type': ['simple']
    #'model_type': ['simple', 'climatenn']
    #'model_type': ['local']
}

defaults = {
    'data_dir': "/storage/data/smooth",
    'volumes_file': '/storage/other/normalizedVolumes.nc',
    'grid_file': '/storage/other/mitgcm-128x64-grid-file.nc',
    'job_dir': "/artifacts",
    'samples': 220
}


parser = argparse.ArgumentParser()
parser.add_argument("--parameters_file", help="")
parser.add_argument("--data-dir", help="", default=defaults['data_dir'])
parser.add_argument("--job-dir", help="", default=defaults['job_dir'])
parser.add_argument("--volumes-file", help="", default=defaults['volumes_file'])
parser.add_argument("--grid-file", help="", default=defaults['grid_file'])
parser.add_argument("--samples", help="", default=defaults['samples'])
parser.add_argument("--print-summaries", help="", action="store_true", default=False)
parser.add_argument("--validation-data", default=None)
parser.add_argument("--download-from", help="", default=None)
parser.add_argument("--upload-to", help="", default=None)
args = parser.parse_args()

parameters = p
if not args.parameters_file is None:
    with open(args.parameters_file, "r") as parameters_file:
        parameters = json.load(parameters_file)

config = {
    'data_dir': args.data_dir,
    'job_dir': args.job_dir,
    'volumes_file': args.volumes_file,
    'grid_file': args.grid_file,
    'samples': args.samples,
    'validation_data': args.validation_data
}

if args.download_from is not None:
    subprocess.check_call(['gsutil', '-m' , 'cp', '-r', args.download_from, '/tmp'])

print(config)
print(os.environ.get('HDF5_USE_FILE_LOCKING'))

if args.print_summaries:
    get_model_summaries(config, parameters)
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

    subprocess.check_call(['gsutil', '-m' , 'cp', '-r', config['job_dir'], args.upload_to])