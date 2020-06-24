from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


def multi_model(model, steps):
    # Allow better summary for this model
    if steps == 1:
        return model

    print(f'Model is of type: {type(model)}')
    input = Input(shape=(15, 64, 128, 1))
    x = input

    for i in range(steps):
        x = model(x)

    output = x

    return Model(inputs=input, outputs=output)
