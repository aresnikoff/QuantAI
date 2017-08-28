from QuantAI import TradingEnv



def sample_algo():

	algo = TradingEnv("sample_pipeline")
	algo.run_backtest("2002-1-1", "2016-1-1", 100000, \
		reward = "sharpe", name = "sample_model", print_iter = 1, n_backtests = 200)


def daily_returns():

	algo = TradingEnv("price", load="daily_returns")
	algo.run_backtest("2002-1-1", "2016-1-1", 100000, \
		reward = "average_daily_returns", name = "daily_returns", print_iter = 25, n_backtests = 150)

def weekly_returns():
	algo = TradingEnv("price", load = "weekly_returns")
	algo.run_backtest("2002-1-1", "2016-1-1", 100000, \
		reward = "average_daily_returns", name = "weekly_returns", print_iter = 25, n_backtests = 150, \
		trade_frequency = "week")

def weekly_returns2():
	algo = TradingEnv("price_100", load = "weekly_returns2")
	algo.run_backtest("2003-1-1", "2016-1-1", 100000, \
		reward = "average_daily_returns", name = "weekly_returns2", print_iter = 25, n_backtests = 150, \
		trade_frequency = "week")	

def monthly_returns():

	algo = TradingEnv("price_30", load = "monthly_returns")
	algo.run_backtest("2002-2-1", "2016-1-1", 100000, \
		reward = "average_daily_returns", name = "monthly_returns", print_iter = 25, n_backtests = 150, \
		trade_frequency = "month")
def percent_returns():

	algo = TradingEnv("percent_returns")
	algo.run_backtest("2002-1-1", "2016-1-1", 100000, \
		reward = "sortino", name = "percent_returns", print_iter = 10, n_backtests = 150, \
		trade_frequency = "week")

def weekly_conv():

	algo = TradingEnv("price_30")
	algo.run_backtest("2002-1-1", "2016-1-1", 100000, \
		reward = "average_daily_returns", name = "weekly_conv", print_iter = 10, n_backtests = 150, \
		trade_frequency = "week")

def main(args):

	algo = TradingEnv(
		pipeline_code = args.pipeline,
		model_code = args.model_code,
		load = args.load,
		n_securities = int(args.n_securities),
		seed = int(args.seed)
	)
	algo.set_agent_params(
		batch_size = int(args.batch_size),
		lr = float(args.lr)
	)
	algo.run_backtest(

			start_date = args.start,
			end_date = args.end,
			capital = int(args.capital),
			name = args.pipeline + "-" + args.model_code,
			print_iter = int(args._print),
			n_backtests = int(args.n_backtests),
			trade_frequency = args.freq

	)

#	algo = TradingEnv("price_30")
#	algo.run_backtest("2002-1-1", "2016-1-1", 100000, \
#		reward = args.reward, name = "cnn_test", print_iter = 2, n_backtests = 150, trade_frequency = "week")


__ALGO_LIST__ = {
	
	"sample_algo": sample_algo,
	"daily_returns": daily_returns,
	"weekly_returns": weekly_returns,
	"weekly_returns2": weekly_returns2,
	"monthly_returns": monthly_returns,
	"percent_returns": percent_returns,
	"weekly_conv": weekly_conv,
	"main": main

}


def get_algorithm(code):

	return __ALGO_LIST__[code]





