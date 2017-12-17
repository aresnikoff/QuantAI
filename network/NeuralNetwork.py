import random
from train import train_and_score

class NeuralNetwork(object):

    def __init__(self, param_options = None):

        self.accuracy = 0.0
        self.param_options = param_options
        self.params = {}

    def __repr__(self):

        rep = str(self.params)
        rep += "\n"
        rep += "Network accuracy: {:.2f}%\n".format(self.accuracy * 100)
        return rep

    def create_random(self):

        for key in self.param_options:

            self.params[key] = random.choice(self.param_options[key])

    def update_params(self, params):

        self.params = params

    def train(self, data, memory):

        if self.accuracy == 0.0:

            self.accuracy = train_and_score(self.params, data, memory)


