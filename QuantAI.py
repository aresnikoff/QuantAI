from zipline.api import order, record, symbol
import gym

def initialize(context):
    
    env = gym.make('quant_ai-v0')


def handle_data(context, data):
    order(symbol('AAPL'), 10)
    record(AAPL=data.current(symbol('AAPL'), 'price'))


