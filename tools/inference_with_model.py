import argparse
import tensorflow.keras as keras
from src.data import get_training_data, save_as_netcdf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-model', help='Path to model')
parser.add_argument('-num-steps', type=int, help='Number of steps we want to predict. Has to be divisible by step-length')
parser.add_argument('-path', help='Path to NetCDF files')
parser.add_argument('-samples', type=int, default=100)
parser.add_argument('-grid-file', help='Path to grid file')
parser.add_argument('-model-predictions-path', help='Model predictions path')
parser.add_argument('--step-length', type=int, help='Step length the model was trained for', default=1)
parser.add_argument('--calculate-mse', default=False, help='Print and calculate mse', action='store_true')

args = parser.parse_args()

# Load data
x, y = get_training_data(args.path, args.samples, wanted_time_difference=args.step_length)

# Make code compatible with RTX2060
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

model = keras.models.load_model(args.model)

predictions = x
for i in range(args.num_steps):
    predictions = model.predict(predictions)

save_as_netcdf(args.grid_file, args.model_predictions_path, predictions, y)

if args.calculate_mse:
    print(f'MSE: {np.average(np.power((predictions - y), 2))}')