# import logging
# import zipline
# from zipline import run_algorithm
# from zipline.finance import commission
# from zipline.pipeline import Pipeline
# from zipline.api import attach_pipeline, pipeline_output, get_datetime
# import pandas as pd
# import datetime
# import numpy as np
# import os


# from network.agents import DQNAgent
# from core.util import *
# from core.rewards import get_reward
# from data.pipelines import get_pipeline

from trading.strategy import Strategy 

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('QuantAI')
log.setLevel(logging.INFO)


class QuantAI(object):

	def __init__(self, param_dict, debug = False):

		# initialize parameters here
		self.cash = param_dict["cash"]
		pass

	def initialize_agent(self, param_dict):

		pass

	def run_backtest(self, start_date, end_date, cash, portfolios):

		log.info("initializing backtest")
		start_date = param_dict["start_date"]
		end_date = param_dict["end_date"]
		for portfolio in portfolios:
			strategy = Strategy(portfolio)
			strategy.run_algorithm


if __name__ == "__main__":

	test = QuantAI(None)
	test.run_backtest(None)


class OutOfMoney(Exception):
	pass

class TradingEnv(object):

	def __init__(self, pipeline_code, model_code, load = "__DO_NOT_USE__",
		n_securities = 250, structure_code = "three-node", seed = 100, hidden_layers = [64], keep_pos = False):

		self.n_securities = n_securities


		self.pipeline_code = pipeline_code
		factor_pipeline, n_factors = get_pipeline(self.pipeline_code)

		self.factor_pipeline  = factor_pipeline
		self.n_factors = n_factors

		self.NetworkSettings = NetworkSettings(self.n_securities, self.n_factors)
		self.EventSettings = EventSettings()
		self.DataProcessor = DataProcessor()
		self.RiskProcessor = RiskProcessor()

		self.state_size, self.n_output = self.NetworkSettings.get_structure(structure_code)
		self.agent = DQNAgent((self.n_securities, self.n_factors), self.n_output, model_code, hidden_layers)
		if load != "__DO_NOT_USE__":
			self.agent.load(load)
		log.info("initialized trading environment with pipeline: " + pipeline_code + "\n")

		self.lag = 50
		self.replay_frequency = 32
		self.total_steps = 0
		self.print_iter = 100

		self.trade_frequency = "day"
		self.keep_pos = keep_pos

		self.prices = {}
		self.risk_values = {}

		np.random.seed(seed if seed > 0 else np.random.randint(low = 1, high = 1000))

	def set_agent_params(self, batch_size = 32, lr = .001):

		self.agent.set_params(batch_size, lr)


	def _initialize(self, context):

		attach_pipeline(self.factor_pipeline, 'factor_pipeline')

		set_commission(commission.PerShare(cost=0.013, min_trade_cost=1.3))

		algo_dates = self.EventSettings.get_date_rule(self.trade_frequency)
		algo_times = self.EventSettings.get_time_rule(self.trade_frequency)

		schedule_function(self.train_agent, algo_dates["start"], algo_times["start"])
		schedule_function(self.review_performance, algo_dates["end"], algo_times["end"])

		context.state = None
		context.action = None
		context.longs = []
		context.shorts = []
		context.last_value = context.portfolio.portfolio_value
		context.returns = []


	def _handle_data(self, context, data):

		pass

	def _before_trading_start(self, context, data):

		pass

	def clear_positions(self, context, data):

		# Get all positions  
		for stock, pos in context.portfolio.positions.iteritems():
			if data.can_trade(stock):
				order_target_percent(stock, 0)

	def execute_trades(self, context, data,  state, action, confidence, clear = False):

		portfolio_capital = context.portfolio.cash
		positions = context.portfolio.positions

		# clear current positions
		if clear:
			self.clear_positions(context, data)

		# determine longs and shorts

		securities = []
		longs = []
		shorts = []

		for i in range(0, self.n_securities, 1):

			eq = state.iloc[i].name
			pos = action[i]
			
			if pos == -1:

				shorts.append(eq)

			elif pos == 1:

				longs.append(eq)

			securities.append(eq)

		# determine weights

		long_weight = confidence / float(len(longs) + len(shorts))
		short_weight = -long_weight
		long_value = long_weight * portfolio_capital if portfolio_capital > 0 else 0
		short_value = short_weight * portfolio_capital if portfolio_capital > 0 else 0
		#long_weight = 0 if len(longs) < 1 else 0.5 / len(longs)
		#short_weight = 0 if len(shorts) < 1 else -0.5 / len(shorts)

		for security in longs:

			is_held = security in positions

			if not is_held and data.can_trade(security):
				try:
					order_target_value(security, long_value)
					#log.info("Long " + str(security.symbol))
				except Exception as e:
					log.warning("Could not order security: " + str(security) + "\n" + e.message)

		for security in shorts:

			is_held = security in positions

			if not is_held and data.can_trade(security):
				try:
					order_target_value(security, short_value)
					#log.info("Short " + str(security.symbol))

				except Exception as e:
					log.warning("Could not order security: " + str(security) + "\n" + e.message)

		for security in securities:

			is_long = security in longs
			is_short = security in shorts
			is_held = security in positions

			if is_held and not is_long and not is_short:

				if data.can_trade(security):


					order_target_value(security, 0)

		context.longs = longs
		context.shorts = shorts
		context.securities = securities

	def find_held_securities(self, positions, market):


		held_idx = []

		for stock, pos in positions.iteritems():

			universe = market.index.tolist()

			if stock in universe:

				idx = universe.index(stock)
				held_idx.append(idx)

		return np.array(held_idx)

	def shuffle_universe(self, all_idx, held_idx):


		# remove len(held_idx) from all_idx
		universe = np.delete(all_idx, held_idx)
		# shuffle
		np.random.shuffle(universe)
		# take needed values
		universe = universe[:self.n_securities - len(held_idx)]
		# add back held
		universe = np.append(universe, held_idx)
		# shuffle again
		np.random.shuffle(universe)

		if (universe.shape[0] != self.n_securities):

			log.warning("There was an error shuffling the universe \n"\
			 + str(all_idx.shape) + str(len(held_idx)) )

			np.random.shuffle(all_idx)
			universe = all_idx[:self.n_securities]

		universe = universe.astype(int)

		return universe


		

	def train_agent(self, context, data):

		output = pipeline_output('factor_pipeline')
		output.dropna(inplace = True)

		# create array of currently_held securities
		if self.keep_pos:
			held_idx = self.find_held_securities(context.portfolio.positions, output)

			if held_idx.shape[0] >= self.n_securities:

				#log.info("holding too many securities. Clearing positions.")
				self.clear_positions(context, data)

			held_idx = np.array([]) 

		else:
			held_idx = np.array([])
		# create array of output indicies
		all_idx = np.arange(output.shape[0])

		idx = self.shuffle_universe(all_idx, held_idx)

		assert idx.shape[0] == self.n_securities, "There are too many securities in the universe"
		state = output.iloc[idx]

		# initialize states and actions
		context.state = state

		risk = self.RiskProcessor.calculate_risk(state, metric = "volatility")
		state = self.DataProcessor.pre_process(state)

		# for each stock in the universe
		for stock in state.index.values.tolist():

			# store the stock's open price
			if not stock.sid in self.prices:

				self.prices[stock.sid] = data[stock].open_price

			if not stock.sid in self.risk_values:

				self.risk_values[stock.sid] = risk[stock]

		#market_state = state
		#portfolio_state = [\
		#	context.portfolio.portfolio_value, \
		# 	context.portfolio.returns

		#]

		#self.agent.state.set_market_state(state)
		#self.agent.state.set_portfolio_state(portfolio_state)

		# agent makes decisions
		action, confidence = self.agent.act(state)
		context.action = action
		context.confidence = confidence
		# make trades based on agents decisions
		self.execute_trades(context, data, state, action, confidence, clear = not self.keep_pos)


	def review_performance(self, context, data):

		sec_returns = {}
		state = context.state
		action = context.action

		if action is None:
			return

		for j in xrange(state.shape[0]):
			stock = state.iloc[j].name
			sec_id = stock.sid
			start_price = self.prices[sec_id]

			current_price = data[stock].close_price
			sec_return = ((current_price - start_price) / start_price) * 100
			sec_returns[sec_id] = sec_return

		self.agent.remember(state, action, self.risk_values, sec_returns)


		today = str(get_datetime())[:-15]
		returns = get_reward("total_returns")(context)
		value = get_reward("portfolio_value")(context)
		reward = get_reward(self.reward)(context)
		confidence = context.confidence
		#daily_return = ((value / context.last_value) - 1) * 100

		#context.returns.append(daily_return)
		#context.last_value = value

		n_short = len(context.shorts)
		n_long = len(context.longs)

		if (self.total_steps+1) % self.print_iter == 0:

			#template = "[{!s}] sharpe: {:03.2f}\tsortino: {:03.2f}\tdaily return: {:03.2f}%\ttotal returns: {:03.2f}%\tvalue: ${:03,.2f}\tn short: {:n}\tn long: {:n}"\
				#.format(today, sharpe, sortino, daily_return, returns, value, n_short, n_long)
			template = "[{!s}]\ttotal returns: {:03.2f}%\tvalue: ${:03,.2f}\tn short: {:n}\tn long: {:n}\t{:03.2f}%"\
				.format(today, returns, value, n_short, n_long, confidence)
			log.info(template)
			#print(template)
			#longs = sorted(context.longs, key = lambda x: x.symbol, reverse = False)
			#shorts = sorted(context.shorts, key = lambda x: x.symbol, reverse = False)
			#log.info("${:03,.2f}".format(value))
			#log.info("short\t\tlong")
			#log.info("-------------------------------------------")
			#for s, l in zip(shorts, longs):

			#	log.info(str(s.symbol) + "\t\t" + str(l.symbol))
			#log.info("-------------------------------------------\n\n")

		
		#self.agent.remember(state, action, reward, None, sec_returns, value < 0)
		self.prices = {}
		self.risk_values = {}



		# if len(self.portfolios) > 1 and value < (self.capital / (1+(1/len(self.portfolios)))):

		# 	self.portfolios = [value]
		# 	log.info(self.portfolios)

		# elif len(self.portfolios) > 1:

		# 	if int(value // self.capital) > len(self.portfolios):

		# 		new_value = value / (float(len(self.portfolios)) + 1)

		# 		self.portfolios = [new_value] * (len(self.portfolios) + 1)
		# 		log.info(self.portfolios)	
		# 	else:		

		# 		new_value = value / float(len(self.portfolios))

		# 		self.portfolios = [new_value] * len(self.portfolios)
		# else:
		# 	if value > self.capital * 2:
		# 		new_value = value / 2.0
		# 		self.portfolios = [new_value, new_value]
		# 		log.info(self.portfolios)
		# 	else:
		# 		self.portfolios = [value]



		#self.clear_positions(context, data)

		if value > 3000:			

			self.total_steps += 1

			if self.total_steps > self.lag and (self.total_steps+1) % self.replay_frequency == 0:

				self.agent.replay()
		else:

			raise OutOfMoney("The algorithm is out of money")
			self.portfolios = [self.capital]


	def run_backtest(self, start_date, end_date, capital, reward = "sharpe", \
			name = "model", print_iter = 100, n_backtests = 100, trade_frequency = "day"):

		self.print_iter = print_iter
		self.reward = reward
		self.trade_frequency = trade_frequency
		self.capital = capital
		run = 0
		while run < n_backtests:

			self.portfolios = [capital]

			log.info("beginning backtest " + str(run+1) + " / " + str(n_backtests))
			print("beginning backtest " + str(run+1) + " / " + str(n_backtests))

			try: 

				perf = run_algorithm(
					start = pd.Timestamp(start_date, tz = "US/Eastern"),
					end = pd.Timestamp(end_date, tz = "US/Eastern"),
					initialize = self._initialize,
					capital_base = capital,
					handle_data = self._handle_data,
					before_trading_start = self._before_trading_start,
					data_frequency = "daily",
					bundle = "quantopian-quandl"
				)

				if run > 5:
					perf_name = "backtests/performance/" + name + "_" + str(run+1) + ".csv"
					perf.to_csv(perf_name)
					log.info("Algorithm succeeded! Saved performance results to: " + perf_name)
					print("Algorithm succeeded! Saved performance results to: " + perf_name)


			except OutOfMoney as e:

				log.info("Algorithm failed. Ran out of money.")
				print("Algorithm failed. Ran out of money")

			run += 1
			if raw_input("Do you want to save?") == 'y':
				self.agent.save(name)





