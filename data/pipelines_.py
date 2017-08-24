from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import (

	AverageDollarVolume,
	AnnualizedVolatility
	### ADD ZIPLINE FACTORS HERE
)

from factors import *


def func():

	price_filter = USEquityPricing.close.latest >= 5
	universe = price_filter
	rng = PriceRange(window_length = 10)

	columns = {}
	columns["range"] = rng

	return Pipeline(columns=columns, screen = universe), len(columns)


def sample_pipeline():

	price_filter = USEquityPricing.close.latest >= 5
	universe = price_filter

	columns = {}
	columns["Last"] = USEquityPricing.close.latest

	return Pipeline(columns = columns, screen = universe), len(columns)


### CREATE PIPELINE HERE



### ADD PIPELINE CODES HERE

__PIPELINE_LIST__ = {
	
	"sample_pipeline": sample_pipeline,
	"func": func

}

def get_pipeline(code):

	return __PIPELINE_LIST__[code]()








