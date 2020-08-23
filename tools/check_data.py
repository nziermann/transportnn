import numpy as np
import netCDF4 as nc4
import argparse

def read_data(data_file, data_variable):
    sub_file = nc4.Dataset(data_file, "r")
    data = sub_file[data_variable]

    return np.array(data)


def check_values(data):
    data = np.array(data)
    return np.any(data < 0)


def check_mass_conversation(data, volumes):
    mass = data*volumes
    mass = np.sum(mass, axis=(1, 2, 3))

    return np.min(mass), np.max(mass), np.max(mass) - np.min(mass)

parser = argparse.ArgumentParser(description='Check mass of data')
parser.add_argument('--data-file', type=str, help='Path to petsc file')
parser.add_argument('--data-variable', type=str, help='Petsc variable')
parser.add_argument('--volume-file', type=str, help='Path to volume file')
args = parser.parse_args()

data = read_data(args.data_file, args.data_variable)
volumes = read_data(args.volume_file, "DUMMY")
print(f'Es gibt negative Werte: {check_values(data)}')
min_mass, max_mass, max_diff_mass = check_mass_conversation(data, volumes)
print(f'Die minimale Masse ist: {min_mass}')
print(f'Die maximale Masse ist: {max_mass}')
print(f'Die maximale Massenabweichung zwischen Schritten ist: {max_diff_mass}')  