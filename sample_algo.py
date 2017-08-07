import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from QuantAI import Algorithm
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import (

	Returns,
	AverageDollarVolume
)

import numpy as np
import tensorflow as tf


class CloseOnN(CustomFactor):  
    # Define inputs
    inputs = [USEquityPricing.close]
    
    # Set window_length to whatever number of days to lookback as a default
    # in the case where no window_length is given when instantiated.
    # This can also be set/over-ridden as shown below:
    # my_close_on_10 = CloseOnN(window_length = 10)
    
    window_length = 2 
    
    def compute(self, today, assets, out, close):  
        out[:] = close[0]


def initialize(context):
    print("used my own init")


def handle_data(context, data):
	pass
    #order(symbol('AAPL'), 10)
    #record(AAPL=data.current(symbol('AAPL'), 'price'))

def make_pipeline():

	price_filter = USEquityPricing.close.latest >= 5
	high_dollar_volume = AverageDollarVolume(window_length = 1).top(500)
	universe = price_filter & high_dollar_volume

	columns = {}
	columns["Last"] = USEquityPricing.close.latest
	columns["Returns"] = Returns(window_length = 2)
	columns["close_10"] = CloseOnN(window_length = 10)
	columns["close_25"] = CloseOnN(window_length = 25)

	# n_days = 10
	# for i in range(2, n_days + 2):
	# 	print(i)
	# 	name = "close_" + str(i-1)
 #        value = CloseOnN(window_length = i)
 #        print(value)
 #        columns[name] = value
	# print(columns)
	return Pipeline(columns = columns, screen = universe)

def run_strategy(context):
	output = context.output
	#print(data)
	data = context.data_output
	a, allQ = context.session.run([predict, Qout], feed_dict={inputs1: data})
	print(a)
	if np.random.rand(1) < context.epsilon:

		a = context.space.sample()
		#print(a)

	context.actions = a
	context.allQ = allQ

	# random long short neutral
	#actions = np.random.randint(low= -1, high = 2, size = 250)
	
	securities = []
	longs = []
	shorts = []

	for i in range(0, 250, 1):

		eq = data.iloc[i].name
		pos = context.actions[i]
		if pos == -1:

			shorts.append(eq)

		elif pos == 1:

			longs.append(eq)

		securities.append(eq)


	return securities, longs, shorts

def update_strategy(context, reward):

	Q1 = context.session.run(Qout,feed_dict={inputs1:context.data_output})
	#Obtain maxQ' and set our target value for chosen action.
	maxQ1 = np.max(Q1)
	targetQ = context.allQ
	targetQ[0,context.actions] = reward + context.y*maxQ1
	#Train our network using target and predicted Q values
	_,W1 = context.session.run([updateModel,W],feed_dict={inputs1:context.data_output,nextQ:targetQ})


if __name__ == "__main__":

	tf.reset_default_graph()


	#These lines establish the feed-forward part of the network used to choose actions
	inputs1 = tf.placeholder(shape=[250,4],dtype=tf.float32)
	W = tf.Variable(tf.random_uniform([4,3],0,0.01))
	Qout = tf.matmul(inputs1,W)
	predict = tf.argmax(Qout,1)

	#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
	nextQ = tf.placeholder(shape=[250,3],dtype=tf.float32)
	loss = tf.reduce_sum(tf.square(nextQ - Qout))
	trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
	updateModel = trainer.minimize(loss)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:

		sess.run(init)

		print("testing sample algorithm")
		my_algo = Algorithm()
		my_algo.initialize = initialize;
		my_algo.make_pipeline = make_pipeline
		my_algo.run_strategy = run_strategy
		my_algo.update_strategy = update_strategy
		my_algo.session = sess
		my_algo.run_backtest("2005-1-1", "2016-1-1", 1000000)










