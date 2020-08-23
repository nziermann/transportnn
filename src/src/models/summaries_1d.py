import numpy as np
import itertools
from tensorflow.keras.utils import plot_model
from src.models import get_model_1d
from src.models.multi import multi_model
from src.models.summaries import product_dict


def get_model_summaries_1d(config, parameters):
    print("Generating models for summaries")
    data = get_dummy_data_1d()
    parameters_product = product_dict(**parameters)
    models_in_row = config.get('models_in_row', 1)

    for idx, parameters_selection in enumerate(parameters_product):
        model = multi_model(get_model_1d(data, parameters_selection), models_in_row)
        #model = get_model(data, parameters_selection)
        print(parameters_selection)
        model.build((1, 52749, 1))
        model.summary()

        from tensorflow.python.keras.layers import wrappers
        from tensorflow.python.keras.engine import network
        print(f'Network: {type(model)}')
        for layer in model._layers:
            print(f'Layer: {type(layer)}')
            print(f'Layer is wrapper: {isinstance(layer, wrappers.Wrapper)}')
            print(f'Layer is network: {isinstance(layer, network.Network)}')

        plot_model(model, expand_nested=True, to_file=f'{config["job_dir"]}/model_{idx}.png')

def get_dummy_data_1d():
    volumes = np.zeros((1, 52749, 1))

    return volumes
