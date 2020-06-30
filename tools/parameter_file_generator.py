import json
import itertools
import subprocess
import shutil
import argparse
import os
import hashlib

#Rough plan
# 1. Load parameter file
# 2. Create new parameter file for every parameter combination
# 3. Foreach parameter file do:
# 3.1 Check if parameter file was already trained skip if it was
# 3.2 Execute main.py
# 3.3 Copy relevant files from data folder
# 3.4 Add parameter file to copied data folder for later tracking purposes

# This should combat the problem of our hyperparameter test messing up
# Additionally it will have the following benefits:
# 1. We get a model file for every model
# 2. We get the model and validation behaviour for every model
# 3. It should be easier to handle and seperate the logs

def load_parameters(file_path):
    with open(file_path, "r") as parameters_file:
        parameters = json.load(parameters_file)

    return parameters

def write_parameters_to_path(parameters, file_path):
    with open(file_path, "w") as parameters_file:
        parameters_file.write(json.dumps(parameters))

def get_parameter_combinations(parameter_definitions):
    keys = parameter_definitions.keys()
    values = parameter_definitions.values()
    for instance in itertools.product(*values):
        wrapped_instance = map(lambda x: [x], instance)
        yield dict(zip(keys, wrapped_instance))

def train_for_parameter_file(parameter_file, script_file, script_options):
    docker_container = 'nziermann/transportnn/cnn-training'
    python = '/root/miniconda/bin/python3'
    script_call = [
        python, script_file, '--parameters-file', parameter_file
    ]
    script_call.extend(script_options)
    subprocess.check_call(script_call)

def copy_data(src, dst):
    shutil.copytree(src, dst)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-parameter-combinations-file', help='Location of file for parameter combinations')
    parser.add_argument('-src-data-dir')
    parser.add_argument('-dst-data-dir')
    parser.add_argument('--model-1d', action="store_true")
    args = parser.parse_args()

    script_file = '/application/src/models/main.py'

    if args.model_1d:
        script_file = '/application/src/models/main_1d.py'


    parameter_data = load_parameters(args.parameter_combinations_file)
    parameter_combinations = get_parameter_combinations(parameter_data)

    parameter_dst_path = os.path.join(args.src_data_dir, 'parameters.json')
    for parameter_combination in parameter_combinations:
        parameter_hash = hashlib.sha1(json.dumps(parameter_combination).encode('utf-8')).hexdigest()
        parameter_dst_data_dir = os.path.join(args.dst_data_dir, parameter_hash)
        parameter_dst_data_path = os.path.join(parameter_dst_data_dir, 'parameters.json')

        script_options = []

        if 'step_length' in parameter_combination:
            script_options.append('--step-length')
            script_options.append(str(parameter_combination['step_length'][0]))

        if 'models_in_row' in parameter_combination:
            script_options.append('--models-in-row')
            script_options.append(str(parameter_combination['models_in_row'][0]))

        #Skips already trained configurations
        #Allows for taking up stopped training runs
        #Or later extension without double traning
        if os.path.isdir(parameter_dst_data_dir):
            print(f'Skipping parameter combination with hash: {parameter_hash}')
            continue

        print(f'Starting training for parameter combination with hash: {parameter_hash}')
        write_parameters_to_path(parameter_combination, parameter_dst_path)
        train_for_parameter_file(parameter_dst_path, script_file, script_options)
        copy_data(args.src_data_dir, parameter_dst_data_dir)
        write_parameters_to_path(parameter_combination, parameter_dst_data_path)


if __name__ == "__main__":
    main()
