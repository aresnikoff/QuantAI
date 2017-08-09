from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import (

	### EXISTING FACTOR IMPORTS GO HERE

	Returns,
	AverageDollarVolume,
	SimpleMovingAverage
)

from factors import *

### CUSTOM PIPELINES GO HERE

def sample_pipeline():

	price_filter = USEquityPricing.close.latest >= 5
	universe = price_filter

	columns = {}
	columns["Last"] = USEquityPricing.close.latest

	return Pipeline(columns = columns, screen = universe), len(columns)


def recent_prices():

	price_filter = USEquityPricing.close.latest >= 5
	high_dollar_volume = AverageDollarVolume(window_length = 5).top(500)
	universe = price_filter & high_dollar_volume

	columns = {}
	columns["Last"] = USEquityPricing.close.latest
	columns["Returns"] = Returns(window_length = 2)
	columns["Returns_10"] = Returns(window_length = 10)

	for i in range(2, 10):
		name = "close_" + str(i)
		val = CloseOnN(window_length = i)
		columns[name] = val

	return Pipeline(columns = columns, screen = universe), len(columns)

def momentum():
	price_filter = USEquityPricing.close.latest >= 5
	high_dollar_volume = AverageDollarVolume(window_length = 5).top(1500)
	universe = price_filter & high_dollar_volume

	columns = {}
	columns["Last"] = USEquityPricing.close.latest
	columns["Returns"] = Returns(window_length = 2)
	columns["SMA_2"] = SimpleMovingAverage(window_length = 2)
	columns["SMA_10"] = SimpleMovingAverage(window_length = 10)

	return Pipeline(columns = columns, screen = universe), len(columns)


""" 
ADD YOUR PIPELINE TO THE PIPELINE LIST 
IN ORDER TO BRING IT INTO YOUR ALGORITHM
"""

__PIPELINE_LIST__ = {
	
	"sample_pipeline": sample_pipeline,
	"momentum": momentum

}

def get_pipeline(code):

	return __PIPELINE_LIST__[code]()













