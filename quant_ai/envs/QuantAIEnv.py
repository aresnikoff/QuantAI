import gym
from gym import error, spaces, utils
from gym.utils import seeding

class Market(object):
	'''
	The state of the market. Consists of a universe of security data and a set of current positions
	'''
	def __init__(self, universe, positions):
		'''
		Args: 
			universe: security data
			positions: current holdings
		'''
		self.universe = universe
		self.positions = positions

	def act(self, action):

		'''
		executes trades

		Returns: 

			a new Market state
		'''
		universe = get_universe()
		positions = get_positions(action)
		return Market(universe, positions)

	def __repr__(self):

		print(positions)


class QuantAIEnv(gym.Env):

	metadata = {'render.modes': ['human']}

	def __init__(self):
		pass


	def _step(self, action):
		pass

	def _reset(self):
		pass

	def _render(self, mode='human', close=False):
		pass




