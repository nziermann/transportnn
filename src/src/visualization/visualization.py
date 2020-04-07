import glob
import numpy as np
import netCDF4 as nc4
import os
from src.data import get_training_data, get_training_data_1d, save_as_netcdf, convert_to_3d
from keras.models import model_from_json
from src.layers import MassConversation1D, MassConversation3D, LandValueRemoval3D
import matplotlib.pyplot as plt
import itertools

def save_data_for_visualization(best_model, data_dir, samples, grid_file, job_dir):
    x, y = get_training_data(data_dir, samples, wanted_time_difference=1)
    x_2, y_2 = get_training_data(data_dir, samples, wanted_time_difference=2)

    predictions = best_model.predict(x)
    save_as_netcdf(grid_file, f'{job_dir}/model_predictions.nc', predictions, y)

    # Generate data for 2 steps based on the best performing model for one step
    predictions_1 = best_model.predict(x_2)
    predictions_2 = best_model.predict(predictions_1)

    save_as_netcdf(grid_file, f'{job_dir}/model_predictions_2.nc', predictions_2, y_2)

def save_data_for_visualization_1d(best_model, data_dir, samples, grid_file, job_dir):
    x, y = get_training_data_1d(data_dir, samples, wanted_time_difference=1)
    x_2, y_2 = get_training_data_1d(data_dir, samples, wanted_time_difference=2)

    predictions = best_model.predict(x)

    predictions = convert_to_3d(predictions, grid_file)
    y = convert_to_3d(y, grid_file)

    save_as_netcdf(grid_file, f'{job_dir}/model_predictions.nc', predictions, y)

    # Generate data for 2 steps based on the best performing model for one step
    predictions_1 = best_model.predict(x_2)
    predictions_2 = best_model.predict(predictions_1)

    predictions_2 = convert_to_3d(predictions_2, grid_file)
    y_2 = convert_to_3d(y_2, grid_file)

    save_as_netcdf(grid_file, f'{job_dir}/model_predictions_2.nc', predictions_2, y_2)

def get_data_diff(data_dir, model = None, model_prediction = None):
    filenames = glob.glob(f'{data_dir}/*.nc')
    # timesteps = {}
    # data = {}
    max_samples = 0

    for filename in filenames:
        data = nc4.Dataset(filename, "r")
        dummy = data["DUMMY"]
        max_samples += np.shape(dummy)[0] - 1

        x = np.full((max_samples, 15, 64, 128, 1), np.nan)
        y = np.full((max_samples, 15, 64, 128, 1), np.nan)
        current_samples = 0

        _, file_name_relative = os.path.split(filename)
        path = os.path.join(data_dir, "visualization" + file_name_relative)
        new_data = nc4.Dataset(path, "w", format = "NETCDF4")

        for dimname, dim in data.dimensions.items():
            dimlength = len(dim)
            if dimname == 'time':
                dimlength = dimlength - 1
            new_data.createDimension(dimname, dimlength)
            new_data.sync()

        #Predict data for everything where we have the real data
        if model_prediction is None:
            model_prediction = model.predict(dummy[:-1])

        for varname, ncvar in data.variables.items():
            if varname in ["time", "depth", "lat", "lon"]:
                var = new_data.createVariable(varname, ncvar.dtype, ncvar.dimensions, zlib=True, fill_value=-9.e+33)
                attdict = ncvar.__dict__
                var.setncatts(attdict)
                if varname == 'time':
                    var[:] = ncvar[1:]
                else:
                    var[:] = ncvar[:]
                new_data.sync()

        var = new_data.createVariable('original', "f8", ("time", "depth", "lat", "lon",), zlib=True, fill_value=-9.e+33)
        var.unit = 1
        var.description = 'original'
        var[:] = dummy[1:]

        var = new_data.createVariable('model_prediction', "f8", ("time", "depth", "lat", "lon",), zlib=True, fill_value=-9.e+33)
        var.unit = 1
        var.description = 'model_prediction'
        var[:] = model_prediction

        var = new_data.createVariable('diff', "f8", ("time", "depth", "lat", "lon",), zlib=True, fill_value=-9.e+33)
        var.unit = 1
        var.description = 'diff'
        var[:] = dummy[1:] - model_prediction

        new_data.close()

    return None

# This method assumes the same format we use otherwise
# So first step is the timestep
def plot_data(datasets, method, ylabel=None, plot_labels=[]):
    timesteps = np.size(datasets[0], axis=0)
    plot_labels_iter = itertools.chain(plot_labels, itertools.repeat(None))

    for data, plot_label in zip(datasets, plot_labels_iter):
        print("Converting data")
        print(f'Plot label: {plot_label}')
        datapoints = np.full(timesteps, np.nan)
        points = [method(data[i]) for i in range(timesteps)]
        datapoints[:] = points
        plt.plot(datapoints, label=plot_label)

    if plot_labels:
        plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel("Timestep")
    plt.show()

def main():

    filename = "/home/nilsz/git/transportnn/data/downloaded_data/model_predictions_complete_validation_data.nc"
    data = nc4.Dataset(filename, "r")
    datasets = [data['original'], data['model_prediction'], np.abs(data['diff'])]

    #assert not np.any(np.isnan(datasets)), "Contains nan data"

    plot_data(datasets, np.mean, ylabel="Mean validation data", plot_labels=['Original', 'Model', 'Difference'])
    plot_data(datasets, np.ndarray.max, ylabel="Max validation data", plot_labels=['Original', 'Model', 'Difference'])

    filename = "/home/nilsz/git/transportnn/data/downloaded_data/model_predictions_complete.nc"
    data = nc4.Dataset(filename, "r")
    datasets = [data['original'], data['model_prediction'], np.abs(data['diff'])]

    #assert not np.any(np.isnan(datasets)), "Contains nan data"

    plot_data(datasets, np.mean, ylabel="Mean training data", plot_labels=['Original', 'Model', 'Difference'])
    plot_data(datasets, np.ndarray.max, ylabel="Max training data", plot_labels=['Original', 'Model', 'Difference'])


if __name__ == '__main__':
    main()

