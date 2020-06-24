import json
import argparse
import subprocess
import os
import glob
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from src.models import get_model_summaries, train_models, train_split_model

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
        #'model_type': ['local']
    }

    defaults = {
        'data_dir': "/storage/data/3d/smooth",
        'validation_data': "/storage/data/3d/validation",
        'volumes_file': '/storage/other/normalizedVolumes.nc',
        'grid_file': '/storage/other/mitgcm-128x64-grid-file.nc',
        'job_dir': "/artifacts",
        # 'samples': np.inf
        'samples': 220,
        'models_in_row': 1,
        'step_length': 1
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters-file", help="")
    parser.add_argument("--data-dir", help="", default=defaults['data_dir'])
    parser.add_argument("--validation-data", default=defaults['validation_data'])
    parser.add_argument("--models-in-row", type=int, default=defaults['models_in_row'])
    parser.add_argument("--step-length", type=int, default=defaults['step_length'])
    parser.add_argument("--predict-data", default=None)
    parser.add_argument("--job-dir", help="", default=defaults['job_dir'])
    parser.add_argument("--volumes-file", help="", default=defaults['volumes_file'])
    parser.add_argument("--grid-file", help="", default=defaults['grid_file'])
    parser.add_argument("--samples", help="", type=int, default=defaults['samples'])
    parser.add_argument("--split", help="", action="store_true", default=False)
    parser.add_argument("--print-summaries", help="", action="store_true", default=False)
    parser.add_argument("--download-from", help="", default=None)
    parser.add_argument("--upload-to", help="", default=None)
    args = parser.parse_args()

    if not args.parameters_file is None:
        with open(args.parameters_file, "r") as parameters_file:
            parameters = json.load(parameters_file)

    #Add step length and models in row for easier evaluation
    parameters['step_length'] = [args.step_length]
    parameters['models_in_row'] = [args.models_in_row] 

    config = {
        'data_dir': args.data_dir,
        'job_dir': args.job_dir,
        'volumes_file': args.volumes_file,
        'grid_file': args.grid_file,
        'samples': args.samples,
        'validation_data': args.validation_data,
        'predict_data': args.predict_data,
        'models_in_row': args.models_in_row,
        'step_length': args.step_length
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