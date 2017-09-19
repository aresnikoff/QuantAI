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
    
    parser = argparse.ArgumentParser(prog = 'QuantAI', description = "Quant AI Algorithm")

    # backtest params
    backtest_group = parser.add_argument_group("Backtest")
    backtest_group.add_argument('--start', help = "The backtest start date", \
        action = "store", dest = "start", type = str, default = "2002-1-1")
    backtest_group.add_argument('--end', help = "The backtest end date", \
        action = "store", dest = "end", type = str, default = "2016-1-1")
    backtest_group.add_argument('--n_securities', help = "The number of \
        securities to trade", action = "store", dest = "n_securities", type = int,\
         default = "100")


    # trading params
    trading_group = parser.add_argument_group("Trading")
    trading_group.add_argument('--cash', help="Starting cash", \
        action = "store", dest = "cash", type=float, default="100000.00")
    trading_group.add_argument('--pipeline', help ='The pipeline to use', \
        action = "store", dest="pipeline", type = str, default="default")
    trading_group.add_argument('--freq', help = "Trade frequency", \
        action = "store", dest= "freq", type = str,  default="day")

    # agent params
    agent_group = parser.add_argument_group("Agent")
    agent_group.add_argument('--model', help='The neural network to use', \
        action = "store", dest = "model", type = str, default = "default")  
    agent_group.add_argument('--batch', help = "Training batch size", \
        action = "store", dest = "batch_size", type = int, default = "32")
    agent_group.add_argument('--lr', help = "The learning rate",
        action = "store", dest = "lr", type = float, default = ".001")

    util_group = parser.add_argument_group("Util")
    util_group.add_argument('--seed', help = "The random seed", \
        action = "store", dest = "seed", type = int, default = "0")   
        
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



