import numpy as np
from src.models import train_models_1d
import argparse
import json
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def main():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    parameters = {
        'filter_exponent': [2, 4, 6],
        'kernel_size': [3, 5, 7],
        'activation': ['elu', 'relu'],
        #'epochs': [100],
        'epochs': [5],
        'batch_norm': [True, False],
        'optimizer': ['adam'],
        #'normalize_input_data': [False],
        #'normalize_mean_input_data': [True],
        'model': ['climatenn', 'tamila', 'tamila_deep'],
        'mass_normalization': [True, False]
    }

    config = {
        'data_dir': "/storage/data/1d/smooth",
        'validation_dir': "/storage/data/1d/validation",
        'volumes_file': "/storage/other/normalizedVolumes.petsc",
        'samples': 220,
        'step_length': 1,
        'models_in_row': 1
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters-file", help="")
    parser.add_argument("--step-length", help="", default=config['step_length'])
    parser.add_argument("--models-in-row", help="", default=config['models_in_row'])

    args = parser.parse_args()

    if not args.parameters_file is None:
        with open(args.parameters_file, "r") as parameters_file:
            parameters = json.load(parameters_file)

    #Add step length and models in row for easier evaluation
    parameters['step_length'] = [args.step_length]
    parameters['models_in_row'] = [args.models_in_row]

    train_models_1d(config, parameters)


if __name__ == "__main__":
    main()