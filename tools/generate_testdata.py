import numpy as np
from mako.template import Template
import os
import shutil


def clear_output_directory(metos3d_output_dir):
    for root, dirs, files in os.walk(metos3d_output_dir):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

def create_nc_file(current_run, length_of_run, metos3d_output_dir, metos3d_petsc2nc_output_dir, metos3d_petsc2nc, grid_file, petsc2nc_conf):
    #/Users/nilsziermann/.metos3d/metos3d/petsc2nc.py 3d 2880 ~/.metos3d/data/data/mitgcm-128x64-grid-file.nc ~/metos3d/experiments/zero/DUMMY.petsc2nc.conf.yaml ~/metos3d/experiments/zero/DUMMY.nc
    metos3d_petsc2nc_output_file_full = os.path.join(metos3d_petsc2nc_output_dir, f'DUMMY_{current_run}.nc')
    os.chdir("/Users/nilsziermann/.metos3d/metos3d")
    cmd = f'{metos3d_petsc2nc} 3d {length_of_run} {grid_file} {petsc2nc_conf} {metos3d_petsc2nc_output_file_full}'
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

metos3d_input_dir = "/tmp/metos3d/input/"
metos3d_input_file = "DUMMY.petsc"
metos3d_output_dir = "/Users/nilsziermann/.metos3d/metos3d/work/"
metos3d_output_file = "DUMMY.petsc"
metos3d_petsc2nc_output_dir = "/tmp/metos3d/nc/"
metos3d_petsc2nc_output_file = "DUMMY.nc"
#input_file = "/Users/nilsziermann/.metos3d/metos3d/input/test.petsc"
metos3d_location = "/Users/nilsziermann/.metos3d/metos3d"
metos3d_model_simpack = "/Users/nilsziermann/.metos3d/metos3d/metos3d-simpack-zero.exe"
metos3d_petsc2nc = "/Users/nilsziermann/.metos3d/metos3d/petsc2nc.py"
grid_file = "/Users/nilsziermann/.metos3d/data/data/mitgcm-128x64-grid-file.nc"
petsc2nc_conf = "~/metos3d/experiments/zero/DUMMY.petsc2nc.conf.yaml"
config_template_file_location = "/Users/nilsziermann/.metos3d/metos3d/model/ZERO/options/template.ZERO.option.txt"
config_file_location="/tmp/metos3d/adapted.ZERO.option.txt"
spinup_count = 5
#spinup_count = 1
vector_size = 52749
number_of_runs = 10
#number_of_runs = 1
length_of_run = 240
#length_of_run = 1
output_file_prefix = "$03d-"
#output_file_prefix = "$01d-"
normalized_value= 2.17
normalized_volume_file = "/Users/nilsziermann/.metos3d/data/data/TMM/2.8/Geometry/normalizedVolumes.petsc"

#Generate config file
mytemplate = Template(filename=config_template_file_location)
config = mytemplate.render(metos3d_input_dir=metos3d_input_dir, metos3d_input_file=metos3d_input_file, metos3d_output_dir=metos3d_output_dir, metos3d_output_file=metos3d_output_file, length_of_run=length_of_run, output_file_prefix=output_file_prefix, spinup_count=spinup_count)

with open(config_file_location, 'w') as f:
    print(config, file=f)

for i in range(number_of_runs):
    clear_output_directory(metos3d_output_dir)
    data = generate_random_data(vector_size)
    data = normalize_data(data, normalized_value, normalized_volume_file)
    print(get_mass(data, normalized_volume_file))
    write_to_file(os.path.join(metos3d_input_dir, metos3d_input_file), data)
    call_metos3d(metos3d_location, metos3d_model_simpack, config_file_location)
    create_nc_file(i, length_of_run, metos3d_output_dir, metos3d_petsc2nc_output_dir, metos3d_petsc2nc, grid_file, petsc2nc_conf)
