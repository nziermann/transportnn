import numpy as np
from functools import partial
from src.visualization import save_data_for_visualization, save_as_netcdf
from src.data import get_training_data, get_volumes, get_landmask, load_netcdf_data, split_data, combine_data
import itertools
from src.models import cnn


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
    assert not np.any(np.isnan(x_train)), "x_train contains nan data"
    assert not np.any(np.isnan(y_train)), "y_train contains nan data"
    x_val, y_val = get_training_data(config['validation_data'], config['samples'])
    assert not np.any(np.isnan(x_val)), "x_val contains nan data"
    assert not np.any(np.isnan(y_val)), "y_val contains nan data"
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

    num_combinations = len(list(parameter_combinations))

    print(f'Found {num_combinations} different combinations')
    for idx, parameter_combination in enumerate(parameter_combinations, start=1):
        print(f'Training combination {idx}/{num_combinations}')
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
    predictions = np.full((samples - 1, 15, 64, 128, 1), np.nan)
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


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))
