# import logging
# import zipline
# from zipline import run_algorithm
# from zipline.finance import commission
# from zipline.pipeline import Pipeline
# from zipline.api import attach_pipeline, pipeline_output, get_datetime
# import pandas as pd
# import datetime
# import numpy as np
# import os


# from network.agents import DQNAgent
# from core.util import *
# from core.rewards import get_reward
# from data.pipelines import get_pipeline

import numpy as np
import pandas as pd

## zipline imports
from zipline import run_algorithm
from zipline.api import schedule_function, attach_pipeline, pipeline_output


from core.pipelines import get_pipeline
from network.agent import Agent
from trading.strategy import Strategy 

from util.helpers import *
from util.exceptions import OutOfMoney

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('QuantAI')
log.setLevel(logging.INFO)


class QuantAI(object):

  def __init__(self, param_dict, debug = False):

    self.param_dict = param_dict

    # initialize parameters here
    self.cash = param_dict["cash"]
    self.n_securities = param_dict["n_securities"];
    self.pipeline, factor_dict = get_pipeline(param_dict)

    for key in factor_dict:
        param_dict[key + "_factors"] = factor_dict[key]

    # init algo step trackers
    self.total_steps = 0
    self.print_iter = param_dict["print_iter"]
    self.n_backtests = param_dict["n_backtests"]

    # set learning parameters
    self.lag = param_dict["lag"]
    self.replay_frequency = param_dict["replay_freq"]

    # set trading parameters
    self.trade_frequency = param_dict["trade_freq"]
    process_trade_freq(self.trade_frequency, param_dict)

    self.keep_positions = param_dict["keep_positions"]

    # get the agent
    self.agent = Agent(param_dict)

    # set random seed for the algorithm
    np.random.seed(param_dict["seed"] if param_dict["seed"] > 0 \
        else np.random.randint(low = 1, high = 1000))


    self.strategies = []

    self.create_strategies(param_dict)
    self.run_backtest(param_dict)


  ## OVERALL TRADING STRATEGY

  def initialize_algo(self, context):


    attach_pipeline(self.pipeline, 'factor_pipeline')

    # select new securities to trade every month
    month_date, month_time = get_date_rules("month")
    day_date, day_time = get_date_rules("day")
    schedule_function(self.update_universe, 
                      month_date["start"], 
                      month_time["start"])

    schedule_function(self.view_status, 
                  day_date["end"], 
                  day_time["end"])

    # initialize strategies
    context.strategies = self.strategies

    # initialize universe
    context.universe = []

    for strategy in context.strategies:

        strategy.initialize(context)

  def _before_trading_start(self, context, data):

    pass

  def handle_data(self, context, data):

    # get pipeline output and remove missing data
    market = pipeline_output('factor_pipeline')
    market.dropna(inplace = True)

    for strategy in context.strategies:

      strategy.update_market(market)

  ## DATA HANDLER
  def update_universe(self, context, data):


    market = pipeline_output('factor_pipeline')

    universe = self.random_universe(market)


    for strategy in context.strategies:

      strategy.update_universe(context, data, universe)

  def random_universe(self, market):

    idx = np.arange(market.shape[0])
    np.random.shuffle(idx)

    securities = np.array([eq.symbol for eq in list(market.index)])
    return securities[idx[:self.n_securities]]

  def view_status(self, context, data):

    self.total_steps += 1

    if self.total_steps % self.print_iter == 0:

      for strategy in self.strategies:

        strategy.view_status()

  ## STRATEGY HANDLER
  def create_strategies(self, param_dict):

    cash = param_dict["cash"]
    start = param_dict["start"]
    end = param_dict["end"]

    n_portfolios = param_dict["n_portfolios"]

    log.info("initializing backtest")
    for portfolio in xrange(n_portfolios):

        portfolio_cash = cash // n_portfolios

        strategy = Strategy(param_dict, self.agent)

        self.strategies.append(strategy)

  ## BACKTEST HANDLER
  def run_backtest(self, param_dict):

    run = 0
    start_date = param_dict["start"]
    end_date = param_dict["end"]
    while run < self.n_backtests:

        try:

            backtest = run_algorithm(

                start = pd.Timestamp(start_date, tz = "US/Eastern"),
                end = pd.Timestamp(end_date, tz = "US/Eastern"),
                initialize = self.initialize_algo,
                capital_base = self.cash,
                handle_data = self.handle_data,
                before_trading_start = self._before_trading_start,
                data_frequency = "daily",
                bundle = "quantopian-quandl"
            )

            save_q = "Do you want to save performance? (y/n)"
            if (raw_input(save_q) == 'y'):

                perf_name = "backtests/performance/" + name + \
                     "_" + str(run+1) + ".csv"
                perf.to_csv(perf_name)
                log.info("Algorithm succeeded! \
                    Saved performance results to: " + perf_name)
                print("Algorithm succeeded! Saved \
                    performance results to: " + perf_name)


        except OutOfMoney as e:

            log.info("Algorithm failed. Ran out of money.")
            print("Algorithm failed. Ran out of money")
        for strategy in self.strategies:

          strategy.__init__(self.param_dict, self.agent)

        run += 1
        if run % 5 == 0:

            self.agent.save()

if __name__ == "__main__":

  test = QuantAI(None)
  #test.run_backtest(None)

