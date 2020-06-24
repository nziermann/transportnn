import numpy as np
from mako.template import Template
import os
import shutil
import argparse


def clear_output_directory(metos3d_output_dir):
    for root, dirs, files in os.walk(metos3d_output_dir):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

def create_nc_file(current_run, length_of_run, metos3d_location, metos3d_petsc2nc_output_dir, metos3d_petsc2nc, grid_file, petsc2nc_conf):
    #/Users/nilsziermann/.metos3d/metos3d/petsc2nc.py 3d 2880 ~/.metos3d/data/data/mitgcm-128x64-grid-file.nc ~/metos3d/experiments/zero/DUMMY.petsc2nc.conf.yaml ~/metos3d/experiments/zero/DUMMY.nc
    metos3d_petsc2nc_output_file_full = os.path.join(metos3d_petsc2nc_output_dir, f'DUMMY_{current_run}.nc')
    os.chdir(metos3d_location)
    cmd = f'python3 {metos3d_petsc2nc} 3d {length_of_run} {grid_file} {petsc2nc_conf} {metos3d_petsc2nc_output_file_full}'
    print(cmd)
    os.system(cmd)

def generate_random_data(length, lower_boundary=0, upper_boundary=1):
    data = np.random.rand(length)
    data = data * (upper_boundary - lower_boundary)
    data = data - lower_boundary

    return data

def call_metos3d(metos3d_location, metos3d_model_simpack, config_file_location):
    # Metos3D configuration is sensitive to the folder the command is executed from
    os.chdir(metos3d_location)
    cmd = f'{metos3d_model_simpack} {config_file_location}'
    print(cmd)
    os.system(cmd)

def read_PETSC_file(file):
    f = open(file, "rb")
    np.fromfile(f, dtype=">i4", count=1)
    nvec, = np.fromfile(f, dtype=">i4", count=1)
    v = np.fromfile(f, dtype=">f8", count=nvec)
    f.close()

    return v

def write_to_file(filename, data):
    f = open(filename, 'wb+')
    # header
    # petsc vecid 1211214
    np.asarray(1211214, dtype='>i4').tofile(f)
    # vector length
    nvec = data.size
    np.asarray(nvec, dtype='>i4').tofile(f)
    # vector values
    np.asarray(data, dtype='>f8').tofile(f)
    f.close()

def normalize_data(data, normalized_value, normalized_volume_file):
    normalized_volumes = read_PETSC_file(normalized_volume_file)

    normalized_volumes = np.expand_dims(normalized_volumes, axis=0)

    normalized_value = np.full((1,), normalized_value)
    normalized_values = data * normalized_volumes
    normalized_data = data * (normalized_value / np.sum(normalized_values))

    return normalized_data

def get_mass(data, normalized_volume_file):
    normalized_volumes = read_PETSC_file(normalized_volume_file)
    normalized_volumes = np.expand_dims(normalized_volumes, axis=0)

    return np.sum(normalized_volumes * data)

metos3d_inputinput_dir = "/tmp/metos3d/input/"
metos3d_input_file = "DUMMY.petsc"
metos3d_output_dir = "/Users/nilsziermann/.metos3d/metos3d/work/"
metos3d_output_file = "DUMMY.petsc"
metos3d_petsc2nc_output_dir = "/tmp/metos3d/nc/"
metos3d_petsc2nc_output_file = "DUMMY.nc"
#input_file = "/Users/nilsziermann/.metos3d/metos3d/input/test.petsc"
metos3d_location = "/Users/nilsziermann/.metos3d/metos3d"
metos3d_model_simpack = "/Users/nilsziermann/.metos3d/metos3d/metos3d-simpack-zero.exe"
metos3d_petsc2nc = os.path.join(metos3d_location, 'petsc2nc.py')
grid_file = "/Users/nilsziermann/.metos3d/data/data/mitgcm-128x64-grid-file.nc"
petsc2nc_conf = "~/metos3d/experiments/zero/DUMMY.petsc2nc.conf.yaml"
config_template_file_location = "/Users/nilsziermann/.metos3d/metos3d/model/ZERO/options/template.ZERO.option.txt"
config_file_location="/tmp/metos3d/adapted.ZERO.option.txt"
spinup_count = 5
#spinup_count = 1
vector_size = 52749
#number_of_runs = 10
number_of_runs = 1
length_of_run = 240
#length_of_run = 1
output_file_prefix = "$03d-"
#output_file_prefix = "$01d-"
normalized_value= 2.17
normalized_volume_file = "/Users/nilsziermann/.metos3d/data/data/TMM/2.8/Geometry/normalizedVolumes.petsc"

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", help="")
parser.add_argument("--input-file", help="")
parser.add_argument("--output-dir", help="")
parser.add_argument("--output-file", help="")
parser.add_argument("--nc-output-dir", help="")
parser.add_argument("--nc-output-file", help="")
parser.add_argument("--metos-location", help="")
parser.add_argument("--metos-model-simpack", help="")
parser.add_argument("--metos-petsc2nc-conf", help="")
parser.add_argument("--template-config-file-location", help="")
parser.add_argument("--config-file-location", help="")
parser.add_argument("--normalized-volume-file", help="")
parser.add_argument("--grid-file", help="")
parser.add_argument("--spinup-count", type=int, help="")
parser.add_argument("--number-of-runs", type=int, help="")
parser.add_argument("--length-of-run", type=int, help="")
parser.add_argument("--output-file-prefix", help="")
parser.add_argument("--normalized-value", type=float, help="")
parser.add_argument("--spinup-model-simpack", default=None)
parser.add_argument("--spinup-template-config-file-location", default=None)
parser.add_argument("--spinup-config-file-location", default=None)
args = parser.parse_args()

metos3d_input_dir = args.input_dir
metos3d_input_file = args.input_file
metos3d_output_dir = args.output_dir
metos3d_output_file = args.output_file
metos3d_petsc2nc_output_dir = args.nc_output_dir
metos3d_petsc2nc_output_file = args.nc_output_file
metos3d_location = args.metos_location
metos3d_model_simpack = args.metos_model_simpack
metos3d_petsc2nc = os.path.join(metos3d_location, 'petsc2nc.py')
petsc2nc_conf = args.metos_petsc2nc_conf
config_template_file_location = args.template_config_file_location
config_file_location = args.config_file_location
normalized_volume_file = args.normalized_volume_file
grid_file = args.grid_file
spinup_count = args.spinup_count
vector_size = 52749
number_of_runs = args.number_of_runs
length_of_run = args.length_of_run
output_file_prefix = args.output_file_prefix
normalized_value= args.normalized_value
spinup_model_simpack = args.spinup_model_simpack
spinup_template_config_file_location = args.spinup_template_config_file_location
spinup_config_file_location = args.spinup_config_file_location

if spinup_model_simpack:
    spinup_template = Template(filename=spinup_template_config_file_location)
    spinup_config = spinup_template.render(
        spinup_count=spinup_count
    )
    spinup_count = 1

    with open(spinup_config_file_location, 'w') as f:
        print(spinup_config, file=f)

#Generate config file
mytemplate = Template(filename=config_template_file_location)
config = mytemplate.render(
    metos3d_input_dir=metos3d_input_dir,
    metos3d_input_file=metos3d_input_file,
    metos3d_output_dir=metos3d_output_dir,
    metos3d_output_file=metos3d_output_file,
    length_of_run=length_of_run,
    spinup_count=spinup_count,
    output_file_prefix=output_file_prefix)

with open(config_file_location, 'w') as f:
    print(config, file=f)

for i in range(number_of_runs):
    clear_output_directory(metos3d_output_dir)
    if spinup_model_simpack:
        call_metos3d(metos3d_location, spinup_model_simpack, spinup_config_file_location)
    else:
        data = generate_random_data(vector_size)
        data = normalize_data(data, normalized_value, normalized_volume_file)
        print(get_mass(data, normalized_volume_file))
        write_to_file(os.path.join(metos3d_input_dir, metos3d_input_file), data)
    call_metos3d(metos3d_location, metos3d_model_simpack, config_file_location)
    create_nc_file(i, length_of_run, metos3d_location, metos3d_petsc2nc_output_dir, metos3d_petsc2nc, grid_file, petsc2nc_conf)
