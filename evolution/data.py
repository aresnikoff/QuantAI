import numpy as np
import pandas as pd
import math
import random

def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]

def generate_series(start, length, bias = .5):

    price = start
    series = [price]
    for i in range(length - 1):

        price += (random.random()*2 - 1) + bias
        series.append(round(price,2))
    last = price (random.random()*2 - 1) + bias
    increase = 1 if last > start else 0
    return series, increase



def monte_carlo(n, k, max_price = 555):

    #set up empty list to hold our ending values for each simulated price series
    X = []
    y = []
 
    for _ in range(n):
        S = round(random.random()*max_price) + round(random.random(), 2)
        T = 252 #Number of trading days
        mu = random.random() #Return
        vol = random.random() #Volatility
 
        daily_returns=np.random.normal(mu/T, vol/math.sqrt(T), T) + 1
        price_list = [S]
 
        for x in daily_returns:
            price_list.append(price_list[-1]*x)

        i = T
        while(i - k - 1 >= 0):

            today = price_list[i]
            series = price_list[i - k - 1:i - 1]
            if today > series[0]*1.02:
                y.append(2)
            else if today < series*.98:
                y.append(0)
            else:
                y.append(1)
            X.append([scale(x, (0, max_price), (0,1)) for x in series])
            i -= 1
    return X, y

def generate_data(n, k, max_price = 555, train_size = .8):

    X, y = monte_carlo(n, k, max_price)

    n_classes = 3
    batch_size = 128 
    input_shape = (k, )

    train = np.random.rand(len(y)) < train_size

    X = pd.DataFrame(X)
    x_train = X[train].as_matrix()
    x_test = X[~train].as_matrix()

    y = np.array(y)
    y_train = y[train]
    y_test = y[~train]

    return n_classes, batch_size, input_shape, x_train, x_test, y_train, y_test    
