import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from algos.helper import run


if __name__ == "__main__":

	parser = argparse.ArgumentParser(prog = 'QuantAI', description = "Quant AI Algorithm")
	parser.add_argument('--algo', help='The algorithm to run', \
		action = "store", dest = "algo_code")
	parser.add_argument('--model', help='The neural network to use', \
		action = "store", dest = "model_code", default = "cnn")
	parser.add_argument('--reward', help='The reward function to use', \
		action = "store", dest = "reward", default = "sharpe")
	parser.add_argument('--filename', help = "The file to store backtest output", \
		action = "store", dest="filename", default = "_none")


	args = parser.parse_args()

	run(args)






