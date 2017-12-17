import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#from algos.helper import run
from QuantAI import QuantAI


def run(args):

    param_dict = {}
    for arg in vars(args):
        param_dict[arg] = getattr(args, arg)

    QuantAI(param_dict)


# read arguments from the command line
def parse_args():
    
  parser = argparse.ArgumentParser(prog = 'QuantAI',
   description = "Quant AI Algorithm")

  # agent params
  agent_group = parser.add_argument_group("Agent")
  agent_group.add_argument('--model', help='The neural network to use', \
    action = "store", dest = "model", type = str, default = "default")  
  agent_group.add_argument('--batch', help = "Training batch size", \
    action = "store", dest = "batch_size", type = int, default = "32")
  agent_group.add_argument('--lr', help = "The learning rate",
    action = "store", dest = "lr", type = float, default = ".05")
  agent_group.add_argument('--lag', help = "The number of iterations \
    before replay", action = 'store', dest = "lag", type = int,
    default = "100")
  agent_group.add_argument('--replay', help="The number of iterations \
    between replays", action = "store", dest = "replay_freq", type = int, \
    default = "32")
  agent_group.add_argument('--gamma', help = "The discount factor", \
    action = "store", dest = "gamma", type = float, default = ".95")

  # backtest params
  backtest_group = parser.add_argument_group("Backtest")
  backtest_group.add_argument('--start', help = "The backtest start date", \
    action = "store", dest = "start", type = str, default = "2002-1-1")
  backtest_group.add_argument('--end', help = "The backtest end date", \
    action = "store", dest = "end", type = str, default = "2016-1-1")
  backtest_group.add_argument('--n_securities', help = "The number of \
    securities to trade", action = "store", dest = "n_securities",  \
    type = int, default = "100")
  backtest_group.add_argument('--n_backtests', help = "The number of \
    backtests to run", action = "store", dest = "n_backtests", \
    type = int, default = "100")

  pipeline_group = parser.add_argument_group("Pipeline")
  pipeline_group.add_argument("--n_days", help = "The number of days \
    worth of data", action = "store", dest = "n_days", type = int,
    default="10")
  pipeline_group.add_argument('--n_stocks', help = "The number of stocks to \
    use in pipeline", action = "store", dest = "n_stocks", type = int,
    default="500")
  pipeline_group.add_argument('--min_price', help = "The minimum \
    price of securities to trade", action = "store", dest = "min_price",
    type = float, default = "5.0")

  # preprocess params
  preprocess_group = parser.add_argument_group("Preprocess")
  preprocess_group.add_argument("--normal", help = "Normalize the data", \
      action = "store_true", dest = "normalize")
  preprocess_group.add_argument('--pca', help = "Run PCA on the data", \
      action = "store_true", dest = "pca")
  preprocess_group.add_argument('--scale', help = "Scale the data to range [0,1]", \
      action = "store_true", dest = "scale")

  # trading params
  trading_group = parser.add_argument_group("Trading")
  trading_group.add_argument('--cash', help="Starting cash", \
      action = "store", dest = "cash", type=float, default="100000.00")
  trading_group.add_argument('--pipeline', help ='The pipeline to use', \
      action = "store", dest="pipeline", type = str, default="default")
  trading_group.add_argument('--freq', help = "Trade frequency (day, week, month)", \
      action = "store", dest= "trade_freq", type = str,  default="day")
  trading_group.add_argument('--n_portfolios', help = "The number of portfolios", \
      action = "store", dest= "n_portfolios", type = int,  default="1")
  trading_group.add_argument('--cost', help = "The cost per trade", \
      action = "store", dest = "cost", type = float, default = ".013")
  trading_group.add_argument('--min_trade_cost', help = "The minimum trade cost", 
      action = "store", dest = "min_trade_cost", type = float, default = "1.3")
  trading_group.add_argument('-k', help = "If flagged, securities will be not be \
      sold after each 'trade_freq' period", action = "store_true", dest = "keep_positions")


  util_group = parser.add_argument_group("Utility")
  util_group.add_argument('--seed', help = "The random seed", \
      action = "store", dest = "seed", type = int, default = "0")  
  util_group.add_argument('--print', help = "Number iterations to print", \
      action = "store", dest = "print_iter", type = int, default = "10") 
  util_group.add_argument('--name', help = "An identifier for the backtest", \
      action = "store", dest = "name", type = str, default = "__AUTO__")
  
  args = parser.parse_args()

  return args 



if __name__ == "__main__":

    args = parse_args()

    run(args)




# parser = argparse.ArgumentParser(prog = 'QuantAI', description = "Quant AI Algorithm")

#     # backtest params

#     parser.add_argument('--start', help = "The backtest start date", \
#         action = "store", dest = "start", type = str, default = "2002-1-1")
#     parser.add_argument('--end', help = "The backtest end date", \
#         action = "store", dest = "end", type = str, default = "2016-1-1")


#     parser.add_argument('--algo', help='The algorithm to run', \
#         action = "store", dest = "algo_code", default = "main")
#     parser.add_argument('--model', help='The neural network to use', \
#         action = "store", dest = "model_code", default = "default")
#     parser.add_argument('--reward', help='The reward function to use', \
#         action = "store", dest = "reward", default = "sharpe")
#     parser.add_argument('--filename', help = "The file to store backtest output", \
#         action = "store", dest="filename", default = "_none")
#     parser.add_argument('--start', help = 'The backtest start date', \
#         action = "store", dest="start", default="2002-1-1")
#     parser.add_argument('--end', help = "The backtest end date", \
#         action = "store", dest = "end", default="2016-1-1")
#     parser.add_argument('--capital', help="Starting cash", \
#         action = "store", dest = "capital", default="100000")
#     parser.add_argument('--load', help = "The trained model weights to load", \
#         action = "store", dest = "load", default="__DO_NOT_USE__")
#     parser.add_argument('--print', help = "How often the algorithm should print", \
#         action = "store", dest = "_print", default="10")
#     parser.add_argument('--n_securities', help = "The number of securities to trade", \
#         action = "store", dest = "n_securities", default="100")
#     parser.add_argument('--n_backtests', help = "The number of backtests to run", \
#         action = "store", dest= "n_backtests", default="100")
#     parser.add_argument('--freq', help = "Trade frequency", \
#         action = "store", dest= "freq", default="day")
#     parser.add_argument('--batch', help = "Training batch size", \
#         action = "store", dest = "batch_size", default = "32")
#     parser.add_argument('--lr', help = "The learning rate",
#         action = "store", dest = "lr", default = ".001")
#     parser.add_argument('--node', help = "The network node structure", \
#         action = "store", dest = "node", default = "three-node")
#     parser.add_argument('--seed', help = "The random seed", \
#         action = "store", dest = "seed", default = "0")
#     parser.add_argument("--layers", help = "hidden layer structure",
#         action = "store", dest = "hidden_layers", default = "64")



