import numpy as np
import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def volatility(row, use_positive = False):

    changes = np.diff(row) / row[:-1] * 100

    # don't include positive changes

    if not use_positive:

        changes = changes[changes <= 0]

    vol = np.std(changes)

    return vol


def calculate_risk(market, metric = "vol"):

  price_col_names = [x for x in market.columns.tolist() if "price_" in x]
  price_cols = list(reversed(natural_sort(price_col_names)))

  price_matrix = market[price_cols]

  if metric == "vol":

    risk = price_matrix.apply(volatility, axis = 1)

    # normalize values
    risk = (risk - np.min(risk))/(np.max(risk) - np.min(risk))

    return risk







