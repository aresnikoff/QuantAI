from zipline.pipeline import CustomFactor
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import (

    Returns,
    AverageDollarVolume,
    SimpleMovingAverage
)
import numpy as np


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