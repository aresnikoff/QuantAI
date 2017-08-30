import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from algos.helper import run


if __name__ == "__main__":

	parser = argparse.ArgumentParser(prog = 'QuantAI', description = "Quant AI Algorithm")
	parser.add_argument('--algo', help='The algorithm to run', \
		action = "store", dest = "algo_code", default = "main")
	parser.add_argument('--model', help='The neural network to use', \
		action = "store", dest = "model_code", default = "default")
	parser.add_argument('--reward', help='The reward function to use', \
		action = "store", dest = "reward", default = "sharpe")
	parser.add_argument('--filename', help = "The file to store backtest output", \
		action = "store", dest="filename", default = "_none")
	parser.add_argument('--pipeline', help ='The pipeline to use', \
		action = "store", dest="pipeline", default="price_avg")
	parser.add_argument('--start', help = 'The backtest start date', \
		action = "store", dest="start", default="2002-1-1")
	parser.add_argument('--end', help = "The backtest end date", \
		action = "store", dest = "end", default="2016-1-1")
	parser.add_argument('--capital', help="Starting cash", \
		action = "store", dest = "capital", default="100000")
	parser.add_argument('--load', help = "The trained model weights to load", \
		action = "store", dest = "load", default="__DO_NOT_USE__")
	parser.add_argument('--print', help = "How often the algorithm should print", \
		action = "store", dest = "_print", default="10")
	parser.add_argument('--n_securities', help = "The number of securities to trade", \
		action = "store", dest = "n_securities", default="100")
	parser.add_argument('--n_backtests', help = "The number of backtests to run", \
		action = "store", dest= "n_backtests", default="100")
	parser.add_argument('--freq', help = "Trade frequency", \
		action = "store", dest= "freq", default="week")
	parser.add_argument('--batch', help = "Training batch size", \
		action = "store", dest = "batch_size", default = "32")
	parser.add_argument('--lr', help = "The learning rate",
		action = "store", dest = "lr", default = ".001")
	parser.add_argument('--node', help = "The network node structure", \
		action = "store", dest = "node", default = "two-node")
	parser.add_argument('--seed', help = "The random seed", \
		action = "store", dest = "seed", default = "100")
	parser.add_argument("--layers", help = "hidden layer structure",
		action = "store", dest = "hidden_layers", default = "64")

	args = parser.parse_args()

	run(args)






