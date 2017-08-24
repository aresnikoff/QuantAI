from QuantAI import TradingEnv



def sample_algo():

	algo = TradingEnv("sample_pipeline")
	algo.run_backtest("2002-1-1", "2016-1-1", 100000, \
		reward = "sharpe", name = "sample_model", print_iter = 1, n_backtests = 200)

def main(args):

	algo = TradingEnv(pipeline_code = args.pipeline, model_code = args.model_code, \
		load = args.load, n_securities = int(args.n_securities))
	algo.run_backtest(

			start_date = args.start,
			end_date = args.end,
			capital = int(args.capital),
			name = args.pipeline + "-" + args.model_code,
			print_iter = int(args._print),
			n_backtests = int(args.n_backtests),
			trade_frequency = args.freq

		)


__ALGO_LIST__ = {
	
	"sample_algo": sample_algo,
	"main": main

}


def get_algorithm(code):

	return __ALGO_LIST__[code]





