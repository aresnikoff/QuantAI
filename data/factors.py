from zipline.pipeline import CustomFactor
from zipline.pipeline.data import USEquityPricing
import numpy as np
from zipline.pipeline.factors import (

	### EXISTING FACTOR IMPORTS GO HERE
	AverageDollarVolume


)


class PriceRange(CustomFactor):
    """
    Computes the difference between the highest high and the lowest
    low of each over an arbitrary input range.
    """
    inputs = [USEquityPricing.high, USEquityPricing.low]

    def compute(self, today, assets, out, highs, lows):
        out[:] = np.nanmax(highs, axis=0) - np.nanmin(lows, axis=0)
### CUSTOM FACTORS GO HERE

