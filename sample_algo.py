import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from QuantAI import TradingEnv

if __name__ == "__main__":

	algo = TradingEnv("recent-prices")
	algo.run_backtest("2005-1-1", "2016-1-1", 100000, name = "momentum")






