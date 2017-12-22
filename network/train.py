from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K


def compile_model(network, n_classes, input_shape):
    """Compile a sequential model.
    Args:
        network (dict): the parameters of the network
    Returns:
        a compiled network.
    """
    # Get our network parameters.
    n_layers = network['n_layers']
    n_neurons = network['n_neurons']
    activation = network['activation']
    optimizer = network['optimizer']
    K.clear_session()
    model = Sequential()

    # Add each layer.
    for i in range(n_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(n_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(n_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train_and_score(network, dataset, memory):
    """Train the model, return test loss.
    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating
    """
    key = tuple(sorted(network.items()))
    if key in memory:
        return memory[key]


    n_classes, batch_size, input_shape = dataset[:3]
    train, validate, test = dataset[3:]
    x_train, y_train = train
    x_validate, y_validate = validate
    x_test, y_test = test

    model = compile_model(network, n_classes, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=1000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_validate, y_validate),
              callbacks=[EarlyStopping()])

    score = model.evaluate(x_test, y_test, verbose=0)

    acc = score[1]  # 1 = accuracy, 0 = loss

    memory[key] = acc

    return acc
