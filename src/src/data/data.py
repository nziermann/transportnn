import numpy as np
import netCDF4 as nc4
import glob

def split_data(data, axis, num_sub, overlap=True, wraparound=True):
    assert num_sub > 1

    length = np.size(data, axis=axis)
    split_length = length//num_sub

    split_starts = []
    split_ends = []
    first_split_start = 0
    if overlap and wraparound:
        first_split_start = -1
    split_starts.append(first_split_start)

    first_split_end = split_length - 1
    if overlap:
        first_split_end = first_split_end + 1
    split_ends.append(first_split_end)

    for i in range(1, num_sub - 1):
        split_start = i * split_length
        if overlap:
            split_start = split_start - 1
        split_starts.append(split_start)

        split_end = (i+1) *split_length - 1
        if overlap:
            split_end = split_end + 1
        split_ends.append(split_end)

    last_split_start = (num_sub - 1) * split_length
    if overlap and wraparound:
        last_split_start = last_split_start - 1
    split_starts.append(last_split_start)

    last_split_end = length - 1
    if overlap:
        last_split_end = last_split_end + 1
    split_ends.append(last_split_end)

    split_data_array = []
    for i in range(0, num_sub):
        split_data_array.append((np.take(data, range(split_starts[i], split_ends[i]+1), axis=axis, mode='wrap')))

    return split_data_array

# We currently assume no overlap of the data and a specific order found in split
def combine_data(data):
    for sub_data in data:
        print(np.shape(sub_data))

    samples = np.size(data[0], axis=0)
    full_data = np.full((samples, 15, 64, 128), np.nan)
    full_data[:, :, 0:32, 0:64] = data[0]
    full_data[:, :, 0:32, 64:128] = data[1]
    full_data[:, :, 32:64, 0:64] = data[2]
    full_data[:, :, 32:64, 64:128] = data[3]

    return full_data

def get_training_data(data_dir, max_samples_wanted, wanted_time_difference=1):
    filenames = glob.glob(f'{data_dir}/*.nc')
    #timesteps = {}
    #data = {}
    max_samples = 0

    for filename in filenames:
        sub_file = nc4.Dataset(filename, "r")
        dummy = sub_file["DUMMY"]
        max_samples += np.shape(dummy)[0]-wanted_time_difference

    if max_samples_wanted < max_samples:
        max_samples = max_samples_wanted

    x = np.full((max_samples, 15, 64, 128, 1), np.nan)
    y = np.full((max_samples, 15, 64, 128, 1), np.nan)
    current_samples = 0

    for filename in filenames:
        sub_file = nc4.Dataset(filename, "r")
        dummy = sub_file["DUMMY"]
        timesteps = np.shape(dummy)[0]

        for i in range(timesteps-wanted_time_difference):
            x[current_samples, :, :, :, 0] = dummy[i]
            y[current_samples, :, :, :, 0] = dummy[i+wanted_time_difference]

            current_samples = current_samples + 1

            # We can return our data immediately
            if current_samples >= max_samples:
                return x, y

    return x, y

def load_netcdf_data(filename):
    sub_file = nc4.Dataset(filename, "r")
    dummy = sub_file["DUMMY"]

    return dummy

def get_training_data_1d(data_dir, max_samples_wanted, wanted_time_difference=1):
    filenames = glob.glob(f'{data_dir}/*.petsc')
    # timesteps = {}
    # data = {}
    max_samples = 0

    max_samples = len(filenames) - wanted_time_difference
    #for filename in filenames:
    #    sub_file = nc4.Dataset(filename, "r")
    #    dummy = sub_file["DUMMY"]
    #    max_samples += np.shape(dummy)[0] - wanted_time_difference

    if max_samples_wanted < max_samples:
        max_samples = max_samples_wanted

    x = np.full((max_samples, 52749, 1), np.nan)
    y = np.full((max_samples, 52749, 1), np.nan)
    current_samples = 0

    for key, filename in enumerate(filenames):
        with open(filename, "rb") as file:
            np.fromfile(file, dtype=">i4", count=1)
            nvec, = np.fromfile(file, dtype=">i4", count=1)
            v = np.fromfile(file, dtype=">f8", count=nvec)

            if key < max_samples:
                x[key, :, 0] = v

            if key >= wanted_time_difference:
                y[key - wanted_time_difference, :, 0] = v
                current_samples = current_samples + 1

            if current_samples >= max_samples:
                return x, y

    return x, y


def get_volumes(volume_file):
    file = nc4.Dataset(volume_file)
    dummy = file["DUMMY"]
    volumes = np.full((15, 64, 128), np.nan)
    volumes[:] = dummy[:]

    return dummy

def get_landmask(landmask_file):
    file = nc4.Dataset(landmask_file)
    print("Variables")
    grid_mask = file["grid_mask"]
    grid_mask = grid_mask[0, :, :, :]

    land_multiplier = np.full((15, 64, 128), 1)
    land_multiplier[grid_mask.mask] = 0

    return land_multiplier

def get_volumes_1d(volume_file):
    with open(volume_file, "rb") as file:
        np.fromfile(file, dtype=">i4", count=1)
        nvec, = np.fromfile(file, dtype=">i4", count=1)
        v = np.fromfile(file, dtype=">f8", count=nvec)

    return v

def get_training_data_sequence(samples, num_input_length, num_output_length):
    x = np.full((samples, num_input_length, 15, 64, 128, 2), np.nan)
    y = np.full((samples, num_output_length, 15, 64, 128, 2), np.nan)

    for i in range(samples):
        x[i] = age[i : i + num_input_length]
        y[i] = age[i + num_input_length : i + num_input_length + num_output_length]

    #Make data 2 dimensional for now
    #TODO: Remove later
    x_2d = x[:, :, 0, :, :, :]
    y_2d = y[:, :, 0, :, :, :]

    #Sanity check
    #is_finite = np.all(np.isfinite(x_2d)) and np.all(np.isfinite(y_2d))
    #print("Finite:")
    #print(is_finite)

    return x_2d, y_2d

#Arrays are assumed to have 4 dimensions according to (time, depth, lat, lon)
#Assumes array predictions and test data have same dimensions
def save_as_netcdf(grid_file_path, file_path, predictions, test_data):
    grid = nc4.Dataset(grid_file_path, "r")
    new_data = nc4.Dataset(file_path, "w", format="NETCDF4")
    test_data = np.reshape(test_data, (-1, 15, 64, 128))
    predictions = np.reshape(predictions, (-1, 15, 64, 128))

    data_shape = np.shape(predictions)
    new_data.createDimension("time", data_shape[0])
    new_data.createDimension("depth", data_shape[1])
    new_data.createDimension("lat", data_shape[2])
    new_data.createDimension("lon", data_shape[3])
    new_data.sync()

    for varname, ncvar in grid.variables.items():
        if varname in ["time", "depth", "lat", "lon"]:
            var = new_data.createVariable(varname, ncvar.dtype, ncvar.dimensions, zlib=True, fill_value=-9.e+33)
            attdict = ncvar.__dict__
            var.setncatts(attdict)
            if varname == 'time':
                ncvar = np.zeros(data_shape[0])
                ncvar[:] = range(0, data_shape[0])

            var[:] = ncvar[:]
            new_data.sync()

    var = new_data.createVariable('original', "f8", ("time", "depth", "lat", "lon",), zlib=True, fill_value=-9.e+33)
    var.unit = 1
    var.description = 'original'
    var[:] = test_data

    var = new_data.createVariable('model_prediction', "f8", ("time", "depth", "lat", "lon",), zlib=True,
                                  fill_value=-9.e+33)
    var.unit = 1
    var.description = 'model_prediction'
    var[:] = predictions

    var = new_data.createVariable('diff', "f8", ("time", "depth", "lat", "lon",), zlib=True, fill_value=-9.e+33)
    var.unit = 1
    var.description = 'diff'
    var[:] = test_data - predictions

    new_data.close()

def convert_to_3d(data, grid_file_path):
    grid = nc4.Dataset(grid_file_path, "r")
    grid_mask_variable = grid.variables["grid_mask"]
    grid_mask = grid_mask_variable[0, :, :, :]

    nz, ny, nx = grid_mask.shape
    work_array = grid_mask.reshape(nz, ny * nx).transpose().flatten()

    num_steps = np.shape(data)[0]
    data = np.reshape(data, (num_steps, np.shape(data)[1]))
    new_data = np.zeros((num_steps, np.size(work_array)))

    print(np.shape(data))
    print(np.shape(new_data))
    print(np.shape(work_array))
    print(np.shape(new_data[:, ~work_array.mask]))

    new_data[:, ~work_array.mask] = data

    var = np.full((num_steps, 15, 64, 128), np.nan)

    var[:, :, :, :] = new_data.reshape((num_steps, ny * nx, nz)).transpose((0, 2, 1)).reshape((num_steps, nz, ny, nx))

    return var