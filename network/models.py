from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')

import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def default_model(input_dim, output_dim, lr, loss):

	model = Sequential()
	model.add(Dense(24, input_dim = input_dim[0]*input_dim[1], activation = 'relu'))
	model.add(Dense(24, activation='relu'))
	model.add(Dense(output_dim, activation='linear'))
	model.compile(loss=loss, optimizer=Adam(lr=lr))

	return model

def cnn(input_dim, output_dim, lr, loss):
	print(input_dim)
	logger.info(input_dim)
	model = Sequential()
	model.add(Convolution1D(256, 3, input_shape = input_dim, activation='relu'))
	logger.info(model.input_shape)
	model.add(Convolution1D(256, 3, activation='relu'))
	model.add(MaxPooling1D(pool_size = 5))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(output_dim, activation='sigmoid'))
	model.compile(loss=loss, optimizer=Adam(lr=lr))
	return model

def binary(input_dim, output_dim, lr, loss):

	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=input_dim[0]*input_dim[1], kernel_initializer='normal', activation='relu'))
	model.add(Dense(output_dim, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
	return model

def three_layer(input_dim, output_dim, lr, loss):

	model = Sequential()
	model.add(Dense(500, input_dim = input_dim[0]*input_dim[1], activation = 'relu'))
	model.add(Dense(250, activation='relu'))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(output_dim, activation='linear'))
	model.compile(loss=loss, optimizer=Adam(lr=lr))

	return model

__NETWORK_CODES__ = {
	
	"default": default_model,
	"cnn": cnn,
	"binary": binary,
	"three-layer": three_layer


}


def get_neural_network(code, input_dim, output_dim, lr, loss = "mse"):

	f = __NETWORK_CODES__[code]
	return f(input_dim, output_dim, lr, loss)




