from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import numpy as np

import matplotlib.pyplot as plt

toolbox = base.Toolbox()

toolbox.register("binary", random.randint, 0, 1)

# Define data

items = ('map', 'compass', 'water', 'sandwich', 'glucose', 'tin', 'banana',
         'apple', 'cheese', 'beer', 'suntan_cream', 'camera', 't-shirt', 'trousers', 'umbrella',
         'waterproof_trousers', 'waterproof_overclothers', 'note-case', 'sunglasses',
         'towel', 'socks', 'books')

item_number = np.arange(1,23)

# Define values
value = (150, 35, 200, 160, 60, 45, 60, 40, 30, 10, 70, 30, 15, 10, 40, 70, 75, 80, 20, 12, 50, 10)

# Define Weight W <- 0(16,6,8,8,7,6)
weight = (9, 13, 153, 50, 15, 68, 27, 39, 23, 52, 11, 32, 24, 48, 73, 42, 43, 22, 7, 18, 4, 30)

# Define Knapsack Capacity (W)
weight_capacity_threshold = 400




print('Item.   Weight   Value')
for i in range(item_number.shape[0]):
    print('{0}          {1}         {2}\n'.format(item_number[i], weight[i], value[i]))

    solutions_per_pop = 8
    pop_size = (solutions_per_pop, item_number.shape[0])
    print('Population size = {}'.format(pop_size))
    initial_population = np.random.randint(2, size=pop_size)
    initial_population = initial_population.astype(int)
    num_generations = 50
    print('Initial population: \n{}'.format(initial_population))




































































    # fitness function for each chromosome
    def fitness(w, v, L, g):  # weight, value, weight_capacity, chromosome
        score = 0
        score1 = 0
        for i in range(len(w)):
            score = score + np.sum(w[i] * g[i])
        if score > L:
            f = 0
        else:
            for i in range(len(w)):
                score1 = score1 + np.sum(v[i] * g[i])
            f = score1
        return score1, score  # fitness