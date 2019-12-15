import glob
import numpy as np
import netCDF4 as nc4
import os
from src.data import get_training_data, save_as_netcdf, convert_to_3d
from talos.utils.best_model import best_model
from keras.models import model_from_json
from src.layers import MassNormalization1D, MassNormalization3D

def save_data_for_visualization(scan_object, data_dir, samples):
    x, y = get_training_data(data_dir, samples, wanted_time_difference=1)
    x_2, y_2 = get_training_data(data_dir, samples, wanted_time_difference=2)

    best_model_id = best_model(scan_object, 'loss', True)
    predict_object = model_from_json(scan_object.saved_models[best_model_id],
                                     {
                                         'MassNormalization3D': MassNormalization3D
                                     })
    predict_object.set_weights(scan_object.saved_weights[best_model_id])

    predictions = predict_object.predict(x)
    save_as_netcdf("/storage/other/mitgcm-128x64-grid-file.nc", "/artifacts/model_predictions.nc", predictions, y)

    # Generate data for 2 steps based on the best performing model for one step
    predictions_1 = predict_object.predict(x_2)
    predictions_2 = predict_object.predict(predictions_1)

    save_as_netcdf("/storage/other/mitgcm-128x64-grid-file.nc", "/artifacts/model_predictions_2.nc", predictions_2, y_2)

def save_data_for_visualization_1d(scan_object, data_dir, samples):
    x, y = get_training_data(data_dir, samples, wanted_time_difference=1)
    x_2, y_2 = get_training_data(data_dir, samples, wanted_time_difference=2)

    best_model_id = best_model(scan_object, 'loss', True)
    predict_object = model_from_json(scan_object.saved_models[best_model_id],
                                     {
                                         'MassNormalization1D': MassNormalization1D
                                     })
    predict_object.set_weights(scan_object.saved_weights[best_model_id])

    predictions = predict_object.predict(x)

    predictions = convert_to_3d(predictions, "/storage/other/mitgcm-128x64-grid-file.nc")
    y = convert_to_3d(predictions, "/storage/other/mitgcm-128x64-grid-file.nc")

    save_as_netcdf("/storage/other/mitgcm-128x64-grid-file.nc", "/artifacts/model_predictions.nc", predictions, y)

    # Generate data for 2 steps based on the best performing model for one step
    predictions_1 = predict_object.predict(x_2)
    predictions_2 = predict_object.predict(predictions_1)

    predictions = convert_to_3d(predictions, "/storage/other/mitgcm-128x64-grid-file.nc")
    y = convert_to_3d(predictions, "/storage/other/mitgcm-128x64-grid-file.nc")

    save_as_netcdf("/storage/other/mitgcm-128x64-grid-file.nc", "/artifacts/model_predictions_2.nc", predictions_2, y_2)

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