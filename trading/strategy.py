from zipline.api import (
	order, record, symbol, schedule_function,
	order_target_percent, set_commission,
	order_target_value, attach_pipeline,
	pipeline_output, get_datetime 
)
from zipline.finance import commission


import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('QuantAI')
log.setLevel(logging.INFO)

class Strategy(object):

	def __init__(self, cash, pipeline, param_dict):

		log.info("initializing strategy")

		self.name = param_dict["name"]

		self.cost = param_dict["cost"]
		self.min_trade_cost = param_dict["min_trade_cost"]
		
		self.train_date = param_dict["train_date"]
		self.train_time = param_dict["train_time"]

		self.review_date = param_dict["review_date"]
		self.review_time = param_dict["review_time"]

		self.cash = cash
		

	def initialize(self, context):

		attach_pipeline(self.pipeline, 'factor_pipeline')

		set_commission(commission.PerShare(cost=0.013, min_trade_cost=1.3))

		# schedule train and review functions

		schedule_function(self.train_agent, self.train_date, self.train_time)
		schedule_function(self.review_performance, self.review_date, self.review_time)


		# initialize context variables
		context.market = None
		context.longs = []
		context.shorts = []

	def train_agent(self, context, data):

		# get pipeline output and remove missing data
		market = pipeline_output('factor_pipeline')
		market.dropna(inplace = True)

		held_securities = self._get_held(context)

		# create array of output indicies
		all_idx = np.arange(output.shape[0])



	def review_performance(self, context, data):

		pass


	def _shuffle_universe(self, all_idx, held_idx):

		pass

	def _get_held(self, context):

		return context.longs + context.shorts

	def _clear_positions(self, context, data):

		# Get all positions  
		for stock, pos in context.portfolio.positions.iteritems():
			if data.can_trade(stock):
				# set percentage to 0
				order_target_percent(stock, 0)




