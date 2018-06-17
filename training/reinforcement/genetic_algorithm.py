from array import array as _
from jetml.neural_network.feedforward import NeuralNetwork, LGNeuralNetwork, InputLayer, HiddenLayer
from jetm.random import choice, randd, randrun
from jetm.matrix import matrix

class NNGeneticAlgorithm:
    def __init__(self, neural_network_config, mutation_rate, max_population, get_fitness):
        self.population = []
        self.generations = 0
        self.neural_network_config = neural_network_config
        self.mutation_rate = mutation_rate
        self.max_population = max_population
        for p in range(max_population):
            self.population.append(NNGene(LGNeuralNetwork.from_config(self.neural_network_config)))
        self.__get_fitness = get_fitness
    
    def calc_fitness(self):
        fitness = self.__get_fitness([g.neural_network for g in self.population])
        for f in range(len(fitness)):
            self.population[f].fitness = fitness[f]
    
    def get_neural_networks(self):
        nns = []
        for p in self.population:
            nns.append(p.neural_network)
        return nns
    
    def update_fitness(self, fitness):
        for p in range(len(self.population)):
            self.population[p].fitness = fitness[p]
    
    def generate_mating_pool(self):
        self.mating_pool = _("f", [])
        fitness_array = [p.fitness for p in self.population]
        max_fitness = max(fitness_array)
        for f in range(len(fitness_array)):
            fitness = fitness_array[f]
            d = fitness / max_fitness
            n = int(round(d * 100))
            for i in range(n):
                self.mating_pool.append(f)

    def generate_next_population(self):
        new_population = []
        for p in range(self.max_population):
            a = int(choice(self.mating_pool))
            reduced = [m for m in self.mating_pool if m != a]
            if len(reduced) == 0:
                reduced.append(self.mating_pool[0])
            b = int(choice(reduced))
            partnerA = self.population[a]
            partnerB = self.population[b]
            child = partnerA.uniform_crossover(partnerB)
            child.mutate(self.mutation_rate)
            new_population.append(child)
        self.population = new_population
        self.generations += 1

    def run_all(self):
        self.calc_fitness()
        self.generate_mating_pool()
        self.generate_next_population()

class NNGene:
    def __init__(self, neural_network):
        self.fitness = 0
        self.neural_network = neural_network
    
    def mutate(self, mutation_rate):
        for l in range(1, len(self.neural_network.layers)):
            layer = self.neural_network.layers[l]
            for r in range(layer.weights.shape[0]):
                for c in range(layer.weights.shape[1]):
                    layer.weights[r,c] = randrun(mutation_rate, (lambda: randd()), (lambda: layer.weights[r,c])).return_value
            for r in range(layer.biases.shape[0]):
                layer.biases[r,0] = randrun(mutation_rate, (lambda: randd()), (lambda: layer.biases[r,0])).return_value
    
    def uniform_crossover(self, gene):
        a = self.neural_network
        b = gene.neural_network
        new_layers = [InputLayer(a.layers[0].dimensionality)]
        for l in range(1, len(a.layers)):
            a_l = a.layers[l]
            b_l = b.layers[l]
            new_weights = matrix.zeros(*a_l.weights.shape)
            for r in range(a_l.weights.shape[0]):
                for c in range(a_l.weights.shape[1]):
                    new_weights[r,c] = choice((a_l.weights[r,c], b_l.weights[r,c]))
            new_biases = matrix.zeros(*a_l.biases.shape)
            for r in range(a_l.biases.shape[0]):
                new_biases[r,0] = choice((a_l.biases[r,0], b_l.biases[r,0]))
            new_layers.append(HiddenLayer(a.layers[l].dimensionality, new_layers[-1].dimensionality, new_weights, new_biases))
        return NNGene(NeuralNetwork(new_layers, a.activation_functions))

