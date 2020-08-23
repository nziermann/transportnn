import numpy as np
import netCDF4 as nc4
import argparse

parser = argparse.ArgumentParser(description='Remove land of nc file')
parser.add_argument('--data-file', type=str, help='Path to nc file')
parser.add_argument('--dimension', type=str, help='Dimension to remove land from')
parser.add_argument('--land-grid', type=str, help='Path to grid file')
args = parser.parse_args()

file = nc4.Dataset(args.land_grid)
grid_mask = file["grid_mask"]
grid_mask = grid_mask[0, :, :, :]

land_multiplier = np.full((15, 64, 128), 1)
land_multiplier[grid_mask.mask] = 0

data = nc4.Dataset(args.data_file, mode='r+')
data[args.dimension][:] = data[args.dimension] * land_multiplier
data.sync()