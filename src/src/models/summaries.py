import numpy as np
import itertools
from tensorflow.keras.utils import plot_model
from src.models import get_model


def get_model_summaries(config, parameters):
    data = get_dummy_data()
    parameters_product = product_dict(**parameters)

    for idx, parameters_selection in enumerate(parameters_product):
        model = get_model(data, parameters_selection)
        print(parameters_selection)
        model.build((1, 15, 64, 128, 1))
        model.summary()

        from tensorflow.python.keras.layers import wrappers
        from tensorflow.python.keras.engine import network
        print(f'Network: {type(model)}')
        for layer in model._layers:
            print(f'Layer: {type(layer)}')
            print(f'Layer is wrapper: {isinstance(layer, wrappers.Wrapper)}')
            print(f'Layer is network: {isinstance(layer, network.Network)}')

        plot_model(model, expand_nested=True, to_file=f'{config["job_dir"]}/model_{idx}.png')


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
