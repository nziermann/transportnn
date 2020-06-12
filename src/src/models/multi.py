from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


def multi_model(model, steps):
    # Allow better summary for this model
    if steps == 1:
        return model

    print(f'Model is of type: {type(model)}')
    input = Input()
    output = input

    for i in range(steps):
        output = model(output)

    return Model(inputs=input, outputs=output)
