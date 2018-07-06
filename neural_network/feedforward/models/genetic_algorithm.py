from array import array as _
from jetml.neural_network.feedforward import NeuralNetwork, LGNeuralNetwork, InputLayer, HiddenLayer, OutputLayer
from jetmath.random import choice, randd, randrun
from jetmath.matrix import matrix

class GeneticAlgorithm:
    def __init__(self, layers, mutation_rate, max_population, activation_functions=("relu", "sigmoid")):
        self.population = []
        self.top = None
        self.gen = 0
        self.mutation_rate = mutation_rate
        self.max_population = max_population
        for p in range(max_population):
            self.population.append(Gene.from_layer_sizes(layers, activation_functions))

    def update_fitness(self, fitness):
        for p in range(len(self.population)):
            self.population[p].fitness = fitness[p]
    
    def __generate_mating_pool(self):
        self.mating_pool = []
        fitness_array = [p.fitness for p in self.population]
        max_fitness = max(fitness_array)
        for f in range(len(fitness_array)):
            fitness = fitness_array[f]
            d = fitness / max_fitness
            n = int(round(d * 100))
            for i in range(n):
                self.mating_pool.append(f)
    
    def update_top(self):
        temp = self.population[0]
        for i in self.population[1:]:
            if i.fitness > temp.fitness:
                temp = i
        if self.top is None:
            self.top = temp.copy()
        else:
            if temp.fitness > self.top.fitness:
                self.top = temp.copy()

    def natural_selection(self):
        self.update_top()
        self.__generate_mating_pool()
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
        self.gen += 1

class Gene(NeuralNetwork):
    def __init__(self, layers, activation_functions=("relu", "sigmoid")):
        self.fitness = 0
        super().__init__(layers, activation_functions)
    
    def mutate(self, mutation_rate):
        for l in range(1, len(self.layers)):
            layer = self.layers[l]
            for r in range(layer.weights.shape[0]):
                for c in range(layer.weights.shape[1]):
                    layer.weights[r,c] = randrun(mutation_rate, (lambda: randd()), (lambda: layer.weights[r,c])).return_value
            for r in range(layer.biases.shape[0]):
                layer.biases[r,0] = randrun(mutation_rate, (lambda: randd()), (lambda: layer.biases[r,0])).return_value
    
    def uniform_crossover(self, gene):
        a = self
        b = gene
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
        return Gene(new_layers, a.activation_functions)
    
    @staticmethod
    def from_layer_sizes(layers, activation_functions):
        if len(layers) < 2:
            raise NeuralNetwork.InitializationException("A feedforward neural network has to have at least 2 layers")
        generated_layers = []
        generated_layers.append(InputLayer(layers[0]))
        if len(layers) > 2:
            for l in range(1, len(layers)-1):
                generated_layers.append(HiddenLayer(layers[l], layers[l-1]))
        generated_layers.append(OutputLayer(layers[-1], layers[-2]))
        return Gene(generated_layers, activation_functions)
    
    def copy(self):
        new = Gene([l.copy() for l in self.layers], self.activation_functions)
        new.fitness = self.fitness
        return new

