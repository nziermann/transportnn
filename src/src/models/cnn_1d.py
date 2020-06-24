from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, AveragePooling1D, UpSampling1D, BatchNormalization, Input, ZeroPadding1D
import tensorflow.keras.metrics
from src.layers import MassConversation1D
from math import ceil

from tensorflow.python.client import device_lib

# Currently allowed parameters of config are
# filter_exponent
# kernel_size
# pooling_type
# activation
# activation_last
def get_model_1d(data, config):
    model = config.get('model', 'climatenn')
    if model == 'simple':
        return get_simple_cnn(data, config)
    return get_convolutional_autoencoder(data, config)

def get_convolutional_autoencoder(data, config):
    model = config.get('model', 'climatenn')
    if model == 'tamila':
        return get_convolutional_autoencoder_tamila(data, config)

    if model == 'tamila_deep':
        return get_convolutional_autoencoder_tamila_deep(data, config)

    if model == 'climatenn_1d':
        return get_convolutional_autoencoder_climatenn(data, config)

    print(f'"Unknown model type {model} requested')
    exit()

def get_simple_cnn(data, config):
    depth = config.get('depth', 6)
    batch_norm = config.get('batch_norm', False)
    filter_exponent = config.get('filter_exponent', 4)
    filters = int(2 ** filter_exponent)
    kernel_size = config.get('kernel_size', (5, 5, 5))
    activation = config.get('activation', 'relu')

    inputs = 52749
    input_shape = (inputs, 1)

    input = Input(shape=input_shape)
    sub_model = Sequential()

    for i in range(depth):
        if batch_norm:
            sub_model.add(BatchNormalization(input_shape=input_shape))
        sub_model.add(Conv1D(filters, kernel_size, input_shape=input_shape, activation=activation, padding='same'))

    output = sub_model(input)
    model = Model(inputs=input, outputs=output)

    if config.get('mass_normalization', True):
        mass_normalization_layer = MassConversation1D(data)([input, output])
        model = Model(inputs=input, outputs=mass_normalization_layer)

    return model

def get_convolutional_autoencoder_tamila_deep(data, config):
    inputs = 52749

    input = Input(shape=(inputs, 1))

    filter_multiplier = config.get('filter_multiplier', 1)
    activation = config.get('activation', 'relu')

    #Encoding
    sub_model = Sequential()
    # conv_1 = Conv1D(10, 3)(input)
    # average_1 = AveragePooling1D(2)(conv_1)
    # conv_2 = Conv1D(8, 3)(average_1)
    # average_2 = AveragePooling1D(2)(conv_2)
    # conv_3 = Conv1D(6, 3)(average_2)
    # average_3 = AveragePooling1D(2)(conv_3)
    # conv_4 = Conv1D(4, 3)(average_3)
    # average_4 = AveragePooling1D(2)(conv_4)
    # conv_5 = Conv1D(2, 3)(average_4)
    # average_5 = AveragePooling1D(2)(conv_5)
    # conv_6 = Conv1D(1, 3)(average_5)
    #
    # #Recoding
    # zero_1 = ZeroPadding1D(4)(conv_6)
    # conv_7 = Conv1D(1, 3)(zero_1)
    # zero_2 = UpSampling1D(2)(conv_7)
    # conv_8 = Conv1D(2, 3)(zero_2)
    # zero_3 = UpSampling1D(2)(conv_8)
    # conv_9 = Conv1D(4, 3)(zero_3)
    # zero_4 = UpSampling1D(2)(conv_9)
    # conv_10 = ZeroPadding1D(1)(zero_4)
    # zero_5 = Conv1D(6, 3)(conv_10)
    # conv_11 = UpSampling1D(2)(zero_5)
    # zero_6 = Conv1D(8, 3)(conv_11)
    # conv_12 = UpSampling1D(2)(zero_6)
    # zero_7 = ZeroPadding1D(1)(conv_12)
    # conv_13 = Conv1D(10, 3)(zero_7)
    # zero_8 = ZeroPadding1D((1,2))(conv_13)
    # conv_14 = Conv1D(1, 3)(zero_8)

    sub_model.add(Conv1D(ceil(filter_multiplier * 10), 3, activation=activation))
    sub_model.add(AveragePooling1D(2))
    sub_model.add(Conv1D(ceil(filter_multiplier * 8), 3, activation=activation))
    sub_model.add(AveragePooling1D(2))
    sub_model.add(Conv1D(ceil(filter_multiplier * 6), 3, activation=activation))
    sub_model.add(AveragePooling1D(2))
    sub_model.add(Conv1D(ceil(filter_multiplier * 4), 3, activation=activation))
    sub_model.add(AveragePooling1D(2))
    sub_model.add(Conv1D(ceil(filter_multiplier * 2), 3, activation=activation))
    sub_model.add(AveragePooling1D(2))
    sub_model.add(Conv1D(ceil(filter_multiplier * 1), 3, activation=activation))

    # Recoding
    sub_model.add(ZeroPadding1D(4))
    sub_model.add(Conv1D(ceil(filter_multiplier * 1), 3, activation=activation))
    sub_model.add(UpSampling1D(2))
    sub_model.add(Conv1D(ceil(filter_multiplier * 2), 3, activation=activation))
    sub_model.add(UpSampling1D(2))
    sub_model.add(Conv1D(ceil(filter_multiplier * 4), 3, activation=activation))
    sub_model.add(UpSampling1D(2))
    sub_model.add(ZeroPadding1D(1))
    sub_model.add(Conv1D(ceil(filter_multiplier * 6), 3, activation=activation))
    sub_model.add(UpSampling1D(2))
    sub_model.add(Conv1D(ceil(filter_multiplier * 8), 3, activation=activation))
    sub_model.add(UpSampling1D(2))
    sub_model.add(ZeroPadding1D(1))
    sub_model.add(Conv1D(ceil(filter_multiplier * 10), 3, activation=activation))
    sub_model.add(ZeroPadding1D((1, 2)))
    sub_model.add(Conv1D(1, 3))

    output = sub_model(input)

    model = Model(inputs=input, outputs=output)

    if config.get('mass_normalization', True):
        mass_normalization_layer = MassConversation1D(data)([input, output])
        model = Model(inputs=input, outputs=mass_normalization_layer)

    return model


def get_convolutional_autoencoder_tamila(data, config):
    inputs = 52749

    input = Input(shape=(inputs, 1))

    sub_model = Sequential()
    sub_model.add(Conv1D(2, 3, activation='elu', input_shape=(inputs, 1), padding='same'))
    sub_model.add(AveragePooling1D(3))
    sub_model.add(Conv1D(4, 3, activation='elu', padding='same'))
    sub_model.add(AveragePooling1D(3))
    sub_model.add(Conv1D(6, 3, activation='elu', padding='same'))
    sub_model.add(AveragePooling1D(1))
    sub_model.add(Conv1D(8, 3, activation='elu', padding='same'))
    sub_model.add(UpSampling1D(1))
    sub_model.add(Conv1D(8, 3, activation='elu', padding='same'))
    sub_model.add(UpSampling1D(3))
    sub_model.add(Conv1D(10, 3, activation='elu', padding='same'))
    sub_model.add(UpSampling1D(3))
    sub_model.add(Conv1D(1, 3, activation='selu', padding='same'))

    output = sub_model(input)

    model = Model(inputs=input, outputs=output)
    if config.get('mass_normalization', True):
        mass_normalization_layer = MassConversation1D(data)([input, output])
        model = Model(inputs=input, outputs=mass_normalization_layer)

    return model
    
def get_convolutional_autoencoder_climatenn(data, config):
    # TODO Readd higher filter choices as option
    filter_exponent = config.get('filter_exponent', 4)
    filters = int(2 ** filter_exponent)
    filters_2 = int(filters / 2)
    kernel_size = config.get('kernel_size', (5, 5, 5))
    pooling_type = config.get('pooling_type', AveragePooling1D)
    activation = config.get('activation', 'relu')
    activation_last = config.get('activation_last', activation)
    batch_norm = config.get('batch_norm', False)

    inputs = 52749

    input = Input(shape=(inputs, 1))

    sub_model = Sequential()
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv1D(filters, kernel_size, activation=activation, padding='same'))
    sub_model.add(pooling_type(3, padding='same'))
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv1D(filters_2, kernel_size, activation=activation, padding='same'))
    sub_model.add(pooling_type(3, padding='same'))
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv1D(filters_2, kernel_size, activation=activation, padding='same'))
    sub_model.add(pooling_type(1, padding='same'))

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv1D(filters_2, kernel_size, activation=activation, padding='same'))
    sub_model.add(UpSampling1D(1))
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv1D(filters_2, kernel_size, activation=activation, padding='same'))
    sub_model.add(UpSampling1D(3))
    if batch_norm:
        sub_model.add(BatchNormalization())
    sub_model.add(Conv1D(filters, kernel_size, activation=activation, padding='same'))
    sub_model.add(UpSampling1D(3))
    if batch_norm:
        sub_model.add(BatchNormalization())

    sub_model.add(Conv1D(1, kernel_size, activation=activation_last, padding='same'))
    output = sub_model(input)

    model = Model(inputs=input, outputs=output)
    if config.get('mass_normalization', True):
        mass_normalization_layer = MassConversation1D(data)([input, output])
        model = Model(inputs=input, outputs=mass_normalization_layer)
    
    return model


def cnn_1d(data, x_train, y_train, x_val, y_val, params):
    print("Getting model with:")
    print(params)

    model = get_convolutional_autoencoder(data, params)

    print("Got model")

    early_stopping_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=10, patience=5,
                                                            restore_best_weights=True)
    callbacks = [early_stopping_callback]
    callbacks = []

    print("Training model")
    metrics = ['mse', tensorflow.keras.metrics.mape, tensorflow.keras.metrics.mae]
    optimizer = params.get('optimizer', 'adam')
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse', tensorflow.keras.metrics.mape, tensorflow.keras.metrics.mae])

    print(params)
    model.summary()

    out = model.fit(x_train, y_train, epochs=params['epochs'], callbacks=callbacks, validation_data=(x_val, y_val))

    return out, model
