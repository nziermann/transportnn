import numpy as np
from src.data import get_training_data_1d, get_volumes_1d
from src.visualization import save_data_for_visualization_1d
from functools import partial
from src.models import cnn_1d, product_dict


def train_models_1d(config, parameters):
    print("Loading data")
    x_train, y_train = get_training_data_1d(config['data_dir'], config['samples'])
    print("Loaded data")

    print("Loading validation data")
    x_val, y_val = get_training_data_1d(config['validation_dir'], np.inf)
    print("Loaded validation data")

    print("Loading volumes")
    data = np.reshape(get_volumes_1d(config['volumes_file']), (1, 52749, 1))
    print("Loaded volumes")
    cnn_partial = partial(cnn_1d, data)

    best_model = None
    lowest_loss = np.inf
    parameter_combinations = product_dict(parameters)
    num_combinations = len(list(parameter_combinations))

    print(f'Found {num_combinations} different combinations')
    for idx, parameter_combination in enumerate(parameter_combinations, start=1):
        print(f'Training combination {idx}/{num_combinations}')

        model, out = cnn_partial(x_train, y_train, x_val, y_val, parameter_combination)
        model_loss = out.history['loss'][-1]

        if model_loss < lowest_loss:
            best_model, lowest_loss = model, model_loss

    # Save data for later visualization
    save_data_for_visualization_1d(best_model, config['data_dir'], config['samples'])