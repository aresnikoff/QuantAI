import gym
from gym.spaces import prng
import numpy as np
import sys


class Positions(gym.Space):
    def __init__(self, n):
        self.n = n
    def sample(self):
        return prng.np_random.randint(low=-1, high=2, size=self.n)
    def contains(self, x):
        return ((x==-1) | (x==0) | (x==1)).all()
    def to_jsonable(self, sample_n):
        return sample_n.tolist()
    def from_jsonable(self, sample_n):
        return np.array(sample_n)


class Market(gym.Env):

	metadata = {"render.modes": ["human", "ansi"]}

	def __init__(self, base_capital = 10000):

		print("initialize market state")

		trade_map = {

			'long': 1,
			'neutral': 0,
			'short': -1
		}

		self.base_capital = base_capital
		self.portfolio = None
		self.market_data = None


	def _reset(self):

		pass


	def _render(self, mode="human", close=False):

		outfile = StringIO() if mode == 'ansi' else sys.stdout
		outfile.write(repr(self.portfolio) + '\n')
		return outfile

	def _step(self, action):

		self.portfolio.trade(action)





