from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import (

    ### ADD ZIPLINE FACTORS HERE
    AverageDollarVolume,
    Returns,
    SimpleMovingAverage,
    EWMA,
    EWMSTD
)
import numpy as np
from factors import *


def price_pipeline(n_days = 10, n_stocks = 1500, min_price = 5):

    # only trade stocks with price > min_price
    price_filter = USEquityPricing.close.latest > min_price

    # look at stocks with high weekly dollar volume
    high_dollar_volume = AverageDollarVolume(window_length = 5).top(n_stocks)

    # create universe of stocks
    universe = (price_filter & high_dollar_volume)

    # initialize columns
    columns = {}

    # add equity pricing data for n_days
    columns["close_1"] = USEquityPricing.close.latest
    for i in range(2, n_days+1):
        name = "close_" + str(i)
        val = CloseOnN(window_length = i)
        columns[name] = val

    return Pipeline(columns = columns, screen = universe), len(columns)


__PIPELINES__ = {
    
    "default": price_pipeline

}


def get_pipeline(code, n_days = 10, n_stocks = 1500, min_price = 5):

    return __PIPELINES__[code](n_days, n_stocks, min_price);






