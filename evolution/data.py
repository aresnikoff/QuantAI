import numpy as np
import pandas as pd
import random

def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]

def generate_series(start, length, bias = .5):

    price = start
    series = [price]
    for i in xrange(length - 1):

        price += (random.random()*2 - 1) + bias
        series.append(round(price,2))

    increase = 1 if series[-1] > start else 0
    return series, increase


def generate_data(n, k, p = .5, max_price = 555, train_size = .8):

    X = []
    y = []
    for i in xrange(n):

        start_price = round(random.random()*max_price) + round(random.random(), 2)

        if random.random() > p:

            series, increase = generate_series(start_price, k, .5)
            y.append(increase)
        else:
            series, increase = generate_series(start_price, k, -.5)
            y.append(increase)

        X.append([scale(x, (0, max_price), (0,1)) for x in series])

    n_classes = 1
    batch_size = 64
    input_shape = (k, )

    train = np.random.rand(n) < train_size

    X = pd.DataFrame(X)
    x_train = X[train].as_matrix()
    x_test = X[~train].as_matrix()

    y = np.array(y)
    y_train = y[train]
    y_test = y[~train]

    return n_classes, batch_size, input_shape, x_train, x_test, y_train, y_test    
