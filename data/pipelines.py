from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import (

	### ADD ZIPLINE FACTORS HERE
	AverageDollarVolume,
	Returns,
	SimpleMovingAverage
)

from factors import *


def sample_pipeline():

	price_filter = USEquityPricing.close.latest >= 5
	universe = price_filter

	columns = {}
	columns["Last"] = USEquityPricing.close.latest

	return Pipeline(columns = columns, screen = universe), len(columns)


### CREATE PIPELINE HERE

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

def price():
	price_filter = USEquityPricing.close.latest >= 5
	high_dollar_volume = AverageDollarVolume(window_length = 5).top(300)

	universe = price_filter & high_dollar_volume

	columns = {}
	columns["close_1"] = USEquityPricing.close.latest
	for i in range(2, 5):
		name = "close_" + str(i)
		val = CloseOnN(window_length = i)
		columns[name] = val

	return Pipeline(columns = columns, screen = universe), len(columns)

def price_100():
	price_filter = USEquityPricing.close.latest >= 5
	high_dollar_volume = AverageDollarVolume(window_length = 5).top(300)

	universe = price_filter & high_dollar_volume

	columns = {}
	columns["close_1"] = USEquityPricing.close.latest
	for i in range(2, 100):
		name = "close_" + str(i)
		val = CloseOnN(window_length = i)
		columns[name] = val

	return Pipeline(columns = columns, screen = universe), len(columns)

def price_30():
	price_filter = USEquityPricing.close.latest >= 5
	high_dollar_volume = AverageDollarVolume(window_length = 5).top(1500)

	universe = price_filter & high_dollar_volume

	columns = {}
	columns["close_1"] = USEquityPricing.close.latest
	for i in range(2, 30):
		name = "close_" + str(i)
		val = CloseOnN(window_length = i)
		columns[name] = val
	return Pipeline(columns = columns, screen = universe), len(columns)



def price_avg():

	n_days = 30

	price_filter = USEquityPricing.close.latest >= 5
	high_dollar_volume = AverageDollarVolume(window_length = 5).top(1500)

	universe = price_filter & high_dollar_volume

	columns = {}
	columns["close_1"] = USEquityPricing.close.latest
	for i in range(2, n_days):
		name = "close_" + str(i)
		val = CloseOnN(window_length = i)
		columns[name] = val
	columns["normalize"] = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length = n_days)
	return Pipeline(columns = columns, screen = universe), len(columns) - 1


def percent_returns():

	price_filter = USEquityPricing.close.latest >= 5
	high_dollar_volume = AverageDollarVolume(window_length = 30).top(1500)

	universe = high_dollar_volume & price_filter
	columns = {}
	for i in range(2, 30):
		name = "return%_" + str(i - 1)
		val = PercentChangeOnN(window_length = i)
		columns[name] = val
	
	return Pipeline(columns = columns, screen = universe), len(columns)

### ADD PIPELINE CODES HERE

__PIPELINE_LIST__ = {
	
	"sample_pipeline": sample_pipeline,
	"recent_prices": recent_prices,
	"price": price,
	"price_100": price_100,
	"price_30": price_30,
	"percent_returns": percent_returns,
	"price_avg": price_avg

}

def get_pipeline(code):

	return __PIPELINE_LIST__[code]()













