from evolution.data import generate_data
from evolution.EvolutionaryProcess import EvolutionaryProcess
from tqdm import tqdm

def train_networks(networks, dataset, memory):
    """
    Train each network

    Arguments:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    progress = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset, memory)
        progress.update(1)
    progress.close()

def optimize(n_generations, pop_size, param_options, data):
    """
    Optimize a network using evolutionary algorithm

    Arguments:
        n_generations (int): Number of times to evolve the population
        pop_size (int): Population size
        param_options (dict): Possible network parameters
        data (tuple): Data
    """
    process = EvolutionaryProcess(param_options)
    networks = process.create_population(pop_size)
    memory = {}
    # Evolve the generation.
    for i in range(n_generations):
        print("*** Generation {} of {}".format(i+1, n_generations))

        # Train networks
        train_networks(networks, data, memory)

        # calc average accuracy for this generation
        average_accuracy = get_average_accuracy(networks)

        print("Generation average: {:.2f}%".format(average_accuracy * 100))
        print('-'*80)

        # Evolve population
        if i < n_generations - 1:
            networks = process.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks, n = 5)

def get_average_accuracy(networks):
    """
    Get the average accuracy for a population

    Arguments:
        networks (list): List of networks

    Returns:
        avg_accuracy (float): The average accuracy of given networks
    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def print_networks(networks, n = 1):
    """
    Print a list of networks

    Arguments :
        networks (list): The population of networks
        n (int): The number of networks to print
    """
    for network in networks[:n]:
        print(network)

def find_best_model():

    n_generations = 20
    pop_size = 25
    dataset = generate_data(1500, 8)

    param_options = {
        'n_neurons': [3, 4, 5, 10, 64, 128, 256, 512, 768, 1024],
        'n_layers': [1,2,3,4,5,6,7,8],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam']
    }

    optimize(n_generations, pop_size, param_options, dataset)