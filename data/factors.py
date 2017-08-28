from zipline.pipeline import CustomFactor
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import (

	Returns,
	AverageDollarVolume,
	SimpleMovingAverage
)

#This is the initial factor, and I'd like to calculate the moving average of this  
class PercentChange(CustomFactor):  
	"""  
	Calculates the percent change of input over the given window_length.  
	"""  
	def compute(self, today, assets, out, data):  
		out[:] = (data[-1] - data[0]) / data[0]  

class PercentChangeOnN(CustomFactor):

	inputs = [USEquityPricing.close]

	window_length = 2

	def compute(self, today, assets, out, close):

		out[:] = (close[1] - close[0]) / close[0]

class CloseOnN(CustomFactor):
	# Define inputs
	inputs = [USEquityPricing.close]

	# Set window_length to whatever number of days to lookback as a default
	# in the case where no window_length is given when instantiated.
	# This can also be set/over-ridden as shown below:
	# my_close_on_10 = CloseOnN(window_length = 10)

	window_length = 2 

	def compute(self, today, assets, out, close):  
		out[:] = close[0]


