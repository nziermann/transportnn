import numpy as np
from src.models import train_models_1d

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def main():
    p = {
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
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
        'samples': np.inf
    }

    train_models_1d(config, p)


if __name__ == "__main__":
    main()