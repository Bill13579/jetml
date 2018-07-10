import jetmath as jm

ACTIVATION_FUNCTIONS = {
    "relu": {
        "_": jm.nonlin.relu,
        "derivative": jm.nonlin.relu_p
    },
    "leaky-relu": {
        "_": jm.nonlin.leaky_relu,
        "derivative": jm.nonlin.leaky_relu_p
    },
    "sigmoid": {
        "_": jm.nonlin.sigmoid,
        "derivative": jm.nonlin.sigmoid_p
    }
}

NEAT_WEIGHTS_MUTATION_FUNCTIONS = {
    "replace": {
        "_": lambda x : jm.random.randd(),
        "use": 0.10
    },
    "add-random": {
        "_": lambda x : x + jm.random.uniform(0, 0.2),
        "use": 0.45
    },
    "subtract-random": {
        "_": lambda x : x - jm.random.uniform(0, 0.2),
        "use": 0.45
    }
}

