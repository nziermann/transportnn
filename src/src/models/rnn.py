from tensorflow.keras.layers.convolutional_recurrent import ConvLSTM2D
from tensorflow.keras.layers import RepeatVector, Reshape, TimeDistributed, Conv2D
from tensorflow.keras.models import Sequential
import talos


def get_encoder(params):
    model = Sequential()
    model.add(ConvLSTM2D(params['filters'], params['kernel_size'], padding='same', activation=params['activation'], input_shape=(params['input_length'], 64, 128, 2)))

    return model


def get_decoder(params):
    model = Sequential()

    #Repeat multi dimensional form
    model.add(Reshape((64*128*params['filters'],)))
    model.add(RepeatVector(params['output_length']))
    model.add(Reshape((params['output_length'], 64, 128, params['filters'])))

    model.add(ConvLSTM2D(params['filters'], params['kernel_size'], padding='same', activation=params['activation'], return_sequences=True))
    return model


def get_output(params):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(2, params['kernel_size'], padding='same', activation=params['activation'])))
    return model


def get_model(params):
    encoder = get_encoder(params)
    decoder = get_decoder(params)
    output = get_output(params)
    model = Sequential()
    model.add(encoder)
    model.add(decoder)
    model.add(output)

    model.summary()

    return model

def rnn(x_train, y_train, x_val, y_val, params):
    model = get_model(params)

    model.compile(optimizer=params['optimizer'], loss='mape', metrics=['mape'])

    # TODO: Use early stopping after finding out sufficient values
    # early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=10, patience=5, restore_best_weights=True)
    # callbacks = [early_stopping]
    callbacks = []
    out = model.fit(x_train, y_train, epochs=params['epochs'], callbacks=callbacks, validation_data=(x_val, y_val))

    return out, model


# Define hyperparameters that can be optimized and standards for optimization
p = {
    'filters' : [5],
    'kernel_size': [(5,5)],
    'activation': ['relu'],
    'layers_encoder': [1],
    'layers_decoder': [1],
    'layers_output': [1],
    'optimizer': ['adam'],
    'input_length': [2],
    'output_length': [7],
    'epochs': [100]
}

samples = 11*8*30
input_length = 2
output_length = 7


x, y = get_training_data(samples, input_length, output_length)

t = talos.Scan(x=x, y=y, params=p, model=rnn, experiment_name='rnn')

