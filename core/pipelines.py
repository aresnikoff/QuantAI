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

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def factor_dict(columns):

    categories = ["price", "fund", "other"]
    factors = {}
    
    # initialize factor dict
    for cat in categories:
        factors[cat] = 0

    keys = list(columns.keys())

    # go through column names
    for key in keys:

        # get the column category
        match = [cat for cat in categories if cat in key]
        assert len(match) <= 1, "The column name must only fit in one category"

        # count the column in its category
        if len(match) == 0:

            factors["other"] += 1
        else:

            factors[match[0]] += 1

    return factors


def price_pipeline(param_dict):

    n_days = param_dict["n_days"]
    n_stocks = param_dict["n_stocks"]
    min_price = param_dict["min_price"]

    # only trade stocks with price > min_price
    price_filter = USEquityPricing.close.latest > min_price

    # look at stocks with high weekly dollar volume
    high_dollar_volume = AverageDollarVolume(window_length = 5).top(n_stocks)

    # create universe of stocks
    universe = (price_filter & high_dollar_volume)

    # initialize columns
    columns = {}

    # add equity pricing data for n_days
    columns["price_1"] = USEquityPricing.close.latest
    for i in range(2, n_days+1):
        name = "price_" + str(i)
        val = CloseOnN(window_length = i)
        columns[name] = val

    return Pipeline(columns = columns, screen = universe), factor_dict(columns)


__PIPELINES__ = {
    
    "default": price_pipeline

}


def get_pipeline(param_dict):

    code = param_dict["pipeline"]
    return __PIPELINES__[code](param_dict)





