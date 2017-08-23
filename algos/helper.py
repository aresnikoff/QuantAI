import sys
from algorithms import get_algorithm
import logging
import os.path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)


def run_algo(args, file):

	code = args.algo_code
	algo = get_algorithm(code)
	for key, value in vars(args).iteritems():
		logger.info("key: " + key + "   value: " + value)
	logger.info("Running algorithm: " + str(code))
	sys.stdout = open('backtests/' + file, 'w')
	algo(args)

def run(args):

	file_name = args.filename
	if file_name == "_none":
		file_name = args.algo_code + "-" + args.model_code
	filename_taken = os.path.isfile("backtests/" + file_name + ".txt")
	count = 1
	while filename_taken:
		file_name = file_name + "_" + str(count)
		filename_taken = os.path.isfile("backtests/" + file_name + ".txt")
		count += 1
	run_algo(args, file_name)


def ru2n(code):

	backtest_file = raw_input("Enter a backtest file name ('n' to use default): ")

	if backtest_file == "n":

		backtest_file = code + ".txt"
	else:

		if backtest_file[-4:] != ".txt":

			backtest_file += ".txt"

	run_algo(code, backtest_file)