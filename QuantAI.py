import logging
import zipline
from zipline import run_algorithm
from zipline.api import order, record, symbol, schedule_function, order_target_percent
from zipline.pipeline import Pipeline
from zipline.api import attach_pipeline, pipeline_output, get_datetime
from trading.util import Market, Positions
import pandas as pd
from zipline.utils.events import (
    EventManager,
    make_eventrule,
    date_rules,
    time_rules,
    AfterOpen,
    BeforeClose
)
import datetime
import numpy as np


class Algorithm(object):

	# load the environment
	#self.MarketState = MarketState
	#self.Portfolio = Portfolio

	def __init__(self):

		print("initializing QuantAI...")
		self.market = Market()
		self.session = None
		self.y = .99
		self.epsilon = .15


	def _initialize(self, context):

		print("init")
		factor_pipeline = self.make_pipeline()
		attach_pipeline(factor_pipeline, 'factor_pipeline')
		schedule_function(self.rebalance, date_rules.every_day(), time_rules.market_open())
		schedule_function(self.review, date_rules.every_day(), time_rules.market_close())
		context.session = self.session
		context.space = Positions(250)
		context.epsilon = self.epsilon
		context.y = self.y
		context.can_trade = False

	def _handle_data(self, context, data):

		portfolio = context.portfolio


		#order(symbol('AAPL'), 10)
		#record(AAPL=data.current(symbol('AAPL'), 'price'))

	def _before_trading_start(self, context, data):

		#context.output = pipeline_output('factor_pipeline')
		# self.market.market_data = output
		if context.can_trade:
			context.securities, context.longs, context.shorts = self.run_strategy(context)
			context.long_weight, context.short_weight = self.compute_weights(context)




		#_, reward, _, _ = self.market.step(self.strategy(output))


	def rebalance(self, context, data):

		if context.can_trade:

			for security in context.longs:
				if data.can_trade(security):
					order_target_percent(security, context.long_weight)

			for security in context.shorts:
				if data.can_trade(security):
					order_target_percent(security, context.short_weight)
			
			for security in context.securities:

				if not security in context.shorts and not security in context.longs:

					order_target_percent(security, 0)


	def review(self, context, data):

		context.output = pipeline_output('factor_pipeline')

		idx = np.arange(250)
		np.random.shuffle(idx)
		data = context.output.iloc[idx]
		context.data_output = data

		reward = self.calculate_reward(context, data)
		if context.can_trade:
			self.update_strategy(context, reward)
		context.can_trade = True

	def calculate_reward(self, context, data):

		today = str(get_datetime())
		today = today[:-15]

		template = "[{!s}] reward: {:03.2f}%   value: ${:03.2f}".format(today, context.portfolio.returns*100, context.portfolio.portfolio_value)
		print(template)
		return context.portfolio.returns



	def compute_weights(self, context):

		long_weight = 0 if len(context.longs) < 1 else 0.5 / len(context.longs)
		short_weight = 0 if len(context.shorts) < 1 else -0.5 / len(context.shorts)

		return long_weight, short_weight 

	def run_strategy(output):

		raise NotImplementedError("QuantAI requires a custom strategy to be used. ")

	def update_strategy(output):

		raise NotImplementedError("QuantAI requires a custom strategy to be used. ")


	def make_pipeline(self):

		raise NotImplementedError("QuantAI requires a custom pipeline to be used. ")


	def run_backtest(self, start_date, end_date, capital):

		print("starting backtest")
		run_algorithm(
			start = pd.Timestamp(start_date, tz = "US/Eastern"),
			end = pd.Timestamp(end_date, tz = "US/Eastern"),
			initialize = self._initialize,
			capital_base = capital,
			handle_data = self._handle_data,
			before_trading_start = self._before_trading_start,
			data_frequency = "daily",
			bundle = "quantopian-quandl"
		)








