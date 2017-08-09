from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import (

	### ADD ZIPLINE FACTORS HERE
)

from factors import *


def sample_pipeline():

	price_filter = USEquityPricing.close.latest >= 5
	universe = price_filter

	columns = {}
	columns["Last"] = USEquityPricing.close.latest

	return Pipeline(columns = columns, screen = universe), len(columns)


### CREATE PIPELINE HERE



### ADD PIPELINE CODES HERE

__PIPELINE_LIST__ = {
	
	"sample_pipeline": sample_pipeline

}

def get_pipeline(code):

	return __PIPELINE_LIST__[code]()








