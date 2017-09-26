from zipline.utils.events import date_rules, time_rules

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('Helpers')
log.setLevel(logging.INFO)


### Date Helpers
def get_date_rules(freq):

  if freq == "day":

    start_rule = date_rules.every_day()
    end_rule = date_rules.every_day()

  elif freq == "week":

    start_rule = date_rules.week_start()
    end_rule = date_rules.week_end()

  elif freq == "month":

    start_rule = date_rules.month_start()
    end_rule = date_rules.month_end()

  else:

    start_rule = date_rules.every_day()
    end_rule = date_rules.every_day()   

  date_rule = {"start": start_rule, "end": end_rule}
  time_rule = {"start": time_rules.market_open(), "end": time_rules.market_close()}

  return date_rule, time_rule

def process_trade_freq(freq, param_dict):

    date_rule, time_rule = get_date_rules(freq)
    param_dict["train_date"] = date_rule["start"]
    param_dict["train_time"] = time_rule["start"]

    param_dict["review_date"] = date_rule["end"]
    param_dict["review_time"] = time_rule["end"]


    





