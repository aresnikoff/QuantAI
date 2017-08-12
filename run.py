
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from algos.helper import run


if __name__ == "__main__":

	algo_code = sys.argv[1]

	run(algo_code)






