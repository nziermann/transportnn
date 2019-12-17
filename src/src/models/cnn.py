from keras.models import Sequential, Model
from keras.layers import Conv3D, AveragePooling3D, UpSampling3D, BatchNormalization, Input, Lambda
import keras.metrics
import numpy as np
from src.data import get_training_data, get_volumes, get_landmask
from src.layers import MassNormalization3D, LandValueRemoval3D
from functools import partial
from src.visualization import save_data_for_visualization
import talos

# Currently allowed parameters of config are
# filter_exponent
# kernel_size
# pooling_type
# activation
# activation_last
def get_convolutional_autoencoder(data, config):
    #TODO Readd higher filter choices as option
    filter_exponent = config.get('filter_exponent', 4)
    filters = int(2**filter_exponent)
    filters_2 = int(filters/2)
    kernel_size = config.get('kernel_size', (5, 5, 5))
    pooling_type = config.get('pooling_type', AveragePooling3D)
    activation = config.get('activation', 'relu')
    activation_last = config.get('activation_last', activation)
    batch_norm = config.get('batch_norm', False)

    input_shape = (15, 64, 128, 1)

    input = Input(shape=input_shape)
    
    sub_model = Sequential()
    if batch_norm:
        sub_model.add(BatchNormalization(input_shape=input_shape))
    sub_model.add(Conv3D(filters, kernel_size, input_shape=input_shape, activation=activation, padding='same'))
    sub_model.add(pooling_type((1, 2, 2), padding='same'))
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv3D(filters_2, kernel_size, activation=activation, padding='same'))
    sub_model.add(pooling_type((1, 2, 2), padding='same'))
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv3D(filters_2, kernel_size, activation=activation, padding='same'))
    sub_model.add(pooling_type((3, 2, 2), padding='same'))

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv3D(filters_2, kernel_size, activation=activation, padding='same'))
    sub_model.add(UpSampling3D((3, 2, 2)))
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv3D(filters_2, kernel_size, activation=activation, padding='same'))
    sub_model.add(UpSampling3D((1, 2, 2)))
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv3D(filters, kernel_size, activation=activation, padding='same'))
    sub_model.add(UpSampling3D((1, 2, 2)))
    if batch_norm:
        sub_model.add(BatchNormalization())

    # cnn = sub_model(input)
    # output = Conv3D(1, kernel_size, activation=activation_last, padding='same')(cnn)
    sub_model.add(Conv3D(1, kernel_size, activation=activation_last, padding='same'))
    output = sub_model(input)

    if config.get('land_removal', True):
        output = LandValueRemoval3D(data['land'])(output)

    #Add normalization layer for mass
    #Handling it this way we can use the model output directly without needing to respect post-processing steps for evaluation and inference
    # def mass_normalization(volumes, tensors):
    #     input = tensors[0]
    #     output = tensors[1]
    #
    #     # Produces vectors with
    #     #input_mass[i] = mass of sample i in batch
    #     #outpus_mass[i] = mass of output for sample i in batch
    #     input_mass = input * volumes
    #     input_mass = K.sum(input_mass, [1,2,3,4])
    #
    #     output_mass = output * volumes
    #     output_mass = K.sum(output_mass, [1,2,3,4])
    #
    #     # Together with the broadcasting of the vector this produces a matrix with
    #     # i,_,_,_,_ = (input_mass[i]/outpus_mass[i])
    #     # Multiplication with this matrix scales the output for each sample so that the proportions are kept
    #     # and the output mass is the same as the input mass for each sample in the batch
    #     vector = (input_mass/output_mass)[np.newaxis][np.newaxis][np.newaxis][np.newaxis]
    #     vector = K.transpose(vector)
    #     normalized_output = output * vector
    #
    #     return [normalized_output]
    #
    # def mass_normalization_output_shape(input_shapes):
    #     shape1 = list(input_shapes[0])
    #     shape2 = list(input_shapes[1])
    #
    #     assert shape1 == shape2  # else normalization is not possible
    #     return [tuple(shape2)]
    #
    # mass_normalization_partial = partial(mass_normalization, data)
    # update_wrapper(mass_normalization_partial, mass_normalization)
    # mass_normalization_layer = Lambda(mass_normalization_partial, mass_normalization_output_shape)([input, output])

    #mass_normalization_layer = MassNormalization3D(data)([input, output])

    #model = Model(inputs=input, outputs=mass_normalization_layer)

    if config.get('mass_normalization', True):
        output = MassNormalization3D(data['volumes'])([input, output])

    optimizer = config.get('optimizer', 'adam')
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss='mse',  metrics=['mse', keras.metrics.mape, keras.metrics.mae])

    model.summary()
    return model

def cnn(data, x_train, y_train, x_val, y_val, params):
    model = get_convolutional_autoencoder(data, params)

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=10, patience=5,
                                                            restore_best_weights=True)
    callbacks = [early_stopping_callback]
    callbacks = []
    out = model.fit(x_train, y_train, epochs=params['epochs'], callbacks=callbacks, validation_data=(x_val, y_val))

    return out, model


p = {
    'filter_exponent': [4],
    'kernel_size': [(3,3,3)],
    'activation': ['elu'],
    'epochs': [100],
    'batch_norm': [False],
    'optimizer': ['adam'],
    'normalize_input_data': [False],
    #'mass_normalization': [True, False],
    'mass_normalization': [True],
    #'land_removal': [True, False],
    'land_removal': [True],
    'normalize_mean_input_data': [False]
}

data_dir = "/storage/data"
volumes_file = "/storage/other/normalizedVolumes.nc"
grid_file = "/storage/other/mitgcm-128x64-grid-file.nc"
samples = np.inf

print("Loading data")
x, y = get_training_data(data_dir, samples)
print("Loaded data")

print("Loading volumes")
volumes = np.reshape(get_volumes(volumes_file), (1, 15, 64, 128, 1))
print("Loaded volumes")

print("Loading land")
land = np.reshape(get_landmask(grid_file), (1, 15, 64, 128, 1))
print("Loaded land")

data = {
    'volumes': volumes,
    'land': land
}

print("Starting model")
cnn_partial = partial(cnn, data)
#cnn_partial(x, y, x, y, single_params)
scan_object = talos.Scan(x=x, y=y, params=p, model=cnn_partial, experiment_name='cnn', x_val=x, y_val=y, save_weights=True)

save_data_for_visualization(scan_object, data_dir, samples)