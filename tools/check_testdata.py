import netCDF4 as nc4
import numpy as np
import argparse

def read_PETSC_file(file):
    f = open(file, "rb")
    np.fromfile(f, dtype=">i4", count=1)
    nvec, = np.fromfile(f, dtype=">i4", count=1)
    v = np.fromfile(f, dtype=">f8", count=nvec)
    f.close()

    return v

def get_mass(data, normalized_volume_file):
    normalized_volumes = read_PETSC_file(normalized_volume_file)
    normalized_volumes = np.expand_dims(normalized_volumes, axis=0)

    return np.sum(normalized_volumes * data)

normalized_volume_file = "/Users/nilsziermann/.metos3d/data/data/TMM/2.8/Geometry/normalizedVolumes.petsc"
parser = argparse.ArgumentParser(description='Check mass of data')
parser.add_argument('data_file', type=str, help='Path to petsc file')
args = parser.parse_args()

data = read_PETSC_file(args.data_file)

print(data)
print(get_mass(data, normalized_volume_file))