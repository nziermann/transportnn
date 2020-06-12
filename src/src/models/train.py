import numpy as np
from functools import partial
from src.visualization import save_data_for_visualization, save_as_netcdf
from src.data import get_training_data, get_volumes, get_landmask, load_netcdf_data, split_data, combine_data
import itertools
from src.models import get_model
import tensorflow
import datetime
import tensorflow.keras as keras
from tensorboard.plugins.hparams import api as hp
from src.models import multi_model


def train(model, x_train, y_train, x_val, y_val, params):
    print("Getting model with:")
    print(params)

    callbacks = []

    # Define the Keras TensorBoard callback.
    logdir = f'/logs/fit/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    callbacks.append(tensorboard_callback)

    hparams_callback = hp.KerasCallback(logdir, params)
    callbacks.append(hparams_callback)

    early_stopping_callback = keras.callbacks.EarlyStopping(patience=10)
    callbacks.append(early_stopping_callback)

    optimizer = params.get('optimizer', 'adam')
    model.compile(optimizer=optimizer, loss='mse',
                  metrics=[tensorflow.keras.metrics.mse, tensorflow.keras.metrics.mape, tensorflow.keras.metrics.mae])
    out = model.fit(x_train, y_train, epochs=params['epochs'], callbacks=callbacks, validation_data=(x_val, y_val))

    return out, model


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

    parameter_combinations = list(product_dict(**parameters))
    best_model = None
    first_model = None
    lowest_loss = np.inf
    num_combinations = len(parameter_combinations)

    print(f'Found {num_combinations} different combinations')
    for idx, parameter_combination in enumerate(parameter_combinations, start=1):
        print(f'Training combination {idx}/{num_combinations}')

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


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))