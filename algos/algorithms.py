from QuantAI import TradingEnv



def sample_algo():

	algo = TradingEnv("sample_pipeline")
	algo.run_backtest("2002-1-1", "2016-1-1", 100000, \
		reward = "sharpe", name = "sample_model", print_iter = 1, n_backtests = 200)

def main(args):

	algo = TradingEnv("price_30")
	algo.run_backtest("2002-1-1", "2016-1-1", 100000, \
		reward = args.reward, name = "cnn_test", print_iter = 2, n_backtests = 150, trade_frequency = "week")


__ALGO_LIST__ = {
	
	"sample_algo": sample_algo,
	"main": main

}


def get_algorithm(code):

	return __ALGO_LIST__[code]





