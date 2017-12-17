# coding=utf-8
import numpy as np
from zipline.api import (
    order, record, symbol, schedule_function,
    order_target_percent, set_commission,
    order_target_value, attach_pipeline,
    pipeline_output, get_datetime 
)
from zipline.finance import commission
from trading.risk import calculate_risk
from util.exceptions import OutOfMoney, BadOutput

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('QuantAI')
log.setLevel(logging.INFO)


class Strategy(object):

  def __init__(self, param_dict, agent):

    self.name = param_dict["name"]

    self.agent = agent

    self.cost = param_dict["cost"]
    self.min_trade_cost = param_dict["min_trade_cost"]
    
    self.train_date = param_dict["train_date"]
    self.train_time = param_dict["train_time"]

    self.review_date = param_dict["review_date"]
    self.review_time = param_dict["review_time"]


    self.n_securities = param_dict["n_securities"]
    self.n_portfolios = param_dict["n_portfolios"]

    self.cash = param_dict["cash"]/float(self.n_portfolios)

    self.param_dict = param_dict

    # initialize strategy variables
    self.universe = []

    self.market = None
    self.actions = None

    self.longs = []
    self.shorts = []

    self.prices = {}
    self.risk_values = {}
    self.sec_returns = {}

    ## Replay vars
    self.total_steps = 0
    self.lag = param_dict["lag"]
    self.replay_frequency = param_dict["replay_freq"]
        

  def initialize(self, context):

    comm = commission.PerShare(cost=self.cost,
                               min_trade_cost=self.min_trade_cost)
    set_commission(comm)

    # schedule train and review functions

    schedule_function(self.train_agent,
                      self.train_date,
                      self.train_time)
    schedule_function(self.review_performance,
                      self.review_date,
                      self.review_time)

  ## TRADING STRATEGY
  def execute_trades(self, context, data, actions):

    cash = self.cash

    market = self.market


    confidence = actions.confidence
    positions = actions.positions

    # determine longs and shorts
    securities = []
    longs = []
    shorts = []

    # add securities to lists
    for i in xrange(self.n_securities):

      eq = market.iloc[i].name
      pos = positions[i]
      
      if pos == -1:

        shorts.append(eq)

      elif pos == 1:

        longs.append(eq)

      securities.append(eq)

    # determine weights

    held = self.held_securities()

    weight = 1 / float(len(held)) if len(held) > len(securities) else 1/float(len(securities))


    for i in xrange(len(held)):

      security = held[i]
      ticker = security.symbol

      if (not security.symbol in self.universe) or \
         (not security in longs and not security in shorts):

        if data.can_trade(security):

          order_target_percent(security, 0)


    for i in xrange(len(securities)):

      security = securities[i]
      c = confidence[i]

      #if c*weight > .05:
        #log.info(c*weight)
      #  pass

      if security in longs:

        if not self.is_short(security) and data.can_trade(security):

          order_target_percent(security, c*weight)


        elif self.is_short(security) and data.can_trade(security):

          order_target_percent(security, 0)

      elif security in shorts:

        if not self.is_long(security) and data.can_trade(security):

          order_target_percent(security, c*weight)
        
        elif self.is_long(security) and data.can_trade(security):

          order_target_percent(security, 0)       

      else:

        if self.is_held(security) and data.can_trade(security):

          order_target_percent(security, 0)

    self.longs = longs
    self.shorts = shorts
    self.securities = securities
    

  def train_agent(self, context, data):

    if len(self.universe) > 0 and not self.market is None:

      market = self.market
      
      risk = calculate_risk(market)
      self.store_metrics(market, data, risk)
      actions = self.agent.act(market)
      self.actions = actions
      self.execute_trades(context, data, actions)


  def review_performance(self, context, data):

    #log.info(self.prices)
    if not self.market is None and len(self.prices.keys()) > 0:
      market = self.market

      sec_returns = {}
      sec_return_values = {}
      for j in xrange(market.shape[0]):
        stock = market.iloc[j].name
        sec_id = stock.sid
        start_price = self.prices[sec_id] if sec_id in self.prices else 1

        current_price = data[stock].close_price
        sec_return = ((current_price - start_price) / start_price) * 100
        sec_returns[sec_id] = sec_return

        # do i want to look at absolute return
        sec_return_value = current_price - start_price
        sec_return_values[sec_id] = sec_return_value

      self.agent.remember(self.market, self.actions,self.risk_values, sec_returns, sec_return_values)
      self.sec_returns = sec_returns
      self.prices = {}

      self.cash = context.portfolio.portfolio_value/float(self.n_portfolios)



      if self.cash > 3000:
        self.total_steps += 1

        if self.total_steps > self.lag and (self.total_steps+1) % self.replay_frequency == 0:

          self.agent.replay()
      else:


        raise OutOfMoney("The algorithm is out of money")


  ## DATA MANAGEMENT

  def update_market(self, market):


    if len(self.universe) > 0:

      # create array of output indicies
      idx = np.arange(market.shape[0])

      #np.random.shuffle(idx)

      market = market.iloc[idx]

      u_tickers = np.array([ticker for ticker in self.universe])

      m_tickers = np.array([eq.symbol for eq in market.index.values.tolist()])

      indices = np.where(np.in1d(m_tickers, u_tickers))[0]

      while len(indices) < len(self.universe):

        # add placeholder values for the day
        rand_idx = np.random.randint(low = 0, high = market.shape[0])
        if not rand_idx in indices:

          indices = np.append(indices,[rand_idx])

      market = market.iloc[indices]

      self.market = market

  def update_universe(self, context, data, universe):

    self.universe = universe

    self.clear_positions(context, data)

  def store_metrics(self, market, data, risk):

    # for each stock in the market
    for stock in market.index.values.tolist():

      # store the stock's open price
      if not stock.sid in self.prices:

        self.prices[stock.sid] = data[stock].open_price

      if not stock.sid in self.risk_values:

        self.risk_values[stock.sid] = risk[stock]

  ## PORTFOLIO MANAGEMENT

  def held_securities(self):

    return self.longs + self.shorts

  def is_long(self, security):

    return security in self.longs

  def is_short(self, security):

    return security in self.shorts

  def is_held(self, security):

    return security in self.held_securities()

  def clear_positions(self, context, data):

    held_securities = self.held_securities()

    for stock, pos in context.portfolio.positions.iteritems():

      if data.can_trade(stock):

        order_target_percent(stock, 0)

      else:

        log.info("COULD NOT CLEAR!")

    self.longs = []
    self.shorts = []

  ## print

  def view_status(self, context, data):


    if not self.actions is None:
      start_cash = float(self.param_dict["cash"])/float(self.n_portfolios)
      today = str(get_datetime())[:-15]
      value = float(self.cash)
      returns = float(float(self.cash - start_cash)/start_cash * 100)
      #confidence = float(self.actions.confidence)
      n_long = int(len(self.longs))
      n_short = int(len(self.shorts))
      leverage = context.account.leverage

      date_str = "[{!s}]\t"
      ret_str = "total returns: {:03.2f}%\t"
      value_str = "value: ${:03,.2f}\t"
      s_str = "n short: {:n}\t"
      l_str = "n long: {:n}\t"
      conf_str = "confidence: {:03.2f}%"
      lev_str = "leverage: {:03.2f}"

      template = date_str + ret_str + value_str + s_str + l_str + lev_str #+ conf_str
      template = template.format(today, returns, value, n_short, n_long, leverage)#, confidence)


    # template = "[{!s}]\ttotal returns: {:03.2f}%\tvalue: ${:03,.2f}\tn short:{:n}\n long: {:n}\tconfidence:{:03.2f}%"\
    #     .format(today, returns, value, n_short, n_long, float(confidence))
      log.info(template)

      stocks_template = ""
      n_held = len(context.portfolio.positions)
      net_gain = 0

      for eq, pos in context.portfolio.positions.iteritems():

        ticker = eq.symbol
        n_shares = pos.amount
        current_val = pos.last_sale_price * n_shares
        paid_val = pos.cost_basis * n_shares
        net_gain += current_val - paid_val

      #   stock_template = "stock {!s}:\tnet gain: {:03.2f}\t".format(ticker, net_gain)
      #   stocks_template += stock_template
      # log.info(stocks_template)
      sign = "$" if net_gain >= 0 else "-$"
      log.info("n held: {:n}\t net gain: {!s}{:03.2f}\t".format(n_held, sign, abs(net_gain)))
      tickers = ""
      eqs = self.market.index.values.tolist()
      for i in xrange(min(len(eqs), 10)):

        tickers += "{!s} ".format(eqs[i].symbol)

      log.info("Market: " + tickers)


      return self.cash




