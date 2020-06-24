import numpy as np
from src.data import get_training_data_1d, get_volumes_1d
from src.models import cnn_1d, product_dict, get_model_1d, multi_model
import datetime
import tensorflow.keras as keras
from tensorboard.plugins.hparams import api as hp
import tensorflow


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

    optimizer = params['optimizer']
    learning_rate = params['learning_rate']

    if optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        print("Other optimizer currently not supported")
        exit()

    model.compile(optimizer=optimizer, loss='mse',
                  metrics=[tensorflow.keras.metrics.mse, tensorflow.keras.metrics.mape, tensorflow.keras.metrics.mae])
    out = model.fit(x_train, y_train, epochs=params['epochs'], callbacks=callbacks, validation_data=(x_val, y_val))

    return out, model


def train_models_1d(config, parameters):
    step_length = config.get('step_length', 1)
    models_in_row = config.get('models_in_row', 1)

    print("Loading data")
    x_train, y_train = get_training_data_1d(config['data_dir'], config['samples'], step_length)
    assert not np.any(np.isnan(x_train)), "x_train contains nan data"
    assert not np.any(np.isnan(y_train)), "y_train contains nan data"
    print("Loaded data")

    print("Loading validation data")
    x_val, y_val = get_training_data_1d(config['validation_dir'], np.inf, step_length)
    assert not np.any(np.isnan(x_train)), "x_train contains nan data"
    assert not np.any(np.isnan(y_train)), "y_train contains nan data"
    print("Loaded validation data")

    print("Loading volumes")
    data = np.reshape(get_volumes_1d(config['volumes_file']), (1, 52749, 1))
    print("Loaded volumes")

    best_model = None
    lowest_loss = np.inf
    parameter_combinations = list(product_dict(**parameters))
    num_combinations = len(parameter_combinations)

    print(f'Found {num_combinations} different combinations')
    for idx, parameter_combination in enumerate(parameter_combinations, start=1):
        print(f'Training combination {idx}/{num_combinations}')

        model = multi_model(get_model_1d(data, parameter_combination), models_in_row)
        out, model = train(model, x_train, y_train, x_val, y_val, parameter_combination)
        model_loss = out.history['loss'][-1]

        if model_loss < lowest_loss:
            best_model, lowest_loss = model, model_loss

    # Save data for later visualization
    #save_data_for_visualization_1d(best_model, config['data_dir'], config['samples'])