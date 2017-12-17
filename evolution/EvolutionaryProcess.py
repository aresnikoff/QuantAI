import random
from functools import reduce
from operator import add
from network.NeuralNetwork import NeuralNetwork

class EvolutionaryProcess(object):

    def __init__(self, param_options, fit_sr = 0.4, unfit_sr = .1,
                 mutation_rate = .2, ):
        """
        Arguments:
            param_options (dict): Possible network parameters
            fit_sr (float): Fit survival rate
            unfit_sr (float): Unfit survival rate
            mutation_rate (float): Probability of mutation
        """
        self.param_options = param_options
        self.fit_sr = fit_sr
        self.unfit_sr = unfit_sr
        self.mutation_rate = mutation_rate

    def create_population(self, n):
        """Create a population of random networks.
        
        Arguments:
            n (int): Population size

        Returns:
            pop (list): Randomly initialized population
        """
        pop = []
        for _ in range(0, n):

            # random network
            network = NeuralNetwork(self.param_options)
            network.create_random()

            # Add the network to the population
            pop.append(network)

        return pop

    def fitness(self, network):
        return network.accuracy

    def average_fitness(self, pop):
        n = float(len(pop))
        total_fit = reduce(add, (self.fitness(network) for network in pop))
        return total_fit / n

    def breed(self, mother, father, max_children = 4):

        n_children = random.randint(0, max_children)
        children = []
        for _ in range(n_children):

            child = {}

            for param in self.param_options:
                genes = [mother.params[param], father.params[param]]
                child[param] = random.choice(genes)

            network = NeuralNetwork(self.param_options)
            network.update_params(child)

            if self.mutation_rate > random.random():

                network = self.mutate(network)

            children.append(network)
        return children

    def mutate(self, network):
        """
        Randomly mutate one param of the network

        Arguments:
            network (NeuralNetwork): The network parameters to mutate

        Returns:
            network (NeuralNetwork): A randomly mutated network object
        """
        # Choose a random key.
        mutation = random.choice(list(self.param_options.keys()))

        # Mutate one of the params.
        network.params[mutation] = random.choice(self.param_options[mutation])

        return network

    def evolve(self, pop):
        """
        Evolve a population of networks
        
        Args:
            pop (list): The population
        Returns:
            parents (list): The evolved population
        """

        # Get scores for each network
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get number of fit individuals
        n_fit_alive = int(len(graded)*self.fit_sr)

        # The parents are all fit
        parents = graded[:n_fit_alive]

        # Include some unfit parents
        for individual in graded[n_fit_alive:]:
            if self.unfit_sr > random.random():
                parents.append(individual)

        parents_length = len(parents)
        pop_size = len(pop) - parents_length
        children = []

        # Add children, which are bred from two alive individuals
        while len(children) < pop_size:

            # Get random parents
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            # breed if parents are not the same
            if male != female:
                male = parents[male]
                female = parents[female]

                babies = self.breed(male, female)

                # Add each child
                for baby in babies:

                    if len(children) < pop_size:
                        children.append(baby)

        parents.extend(children)

        return parents