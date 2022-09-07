from deap import base
from deap import creator
from deap import tools
from deap import algorithms


import random
import numpy as np

import matplotlib.pyplot as plt

## parameters
chrom_size = 100
population_size = 200
P_CROSSOVER = 0.9
M_MUTATION = 0.1
MAX_GENERATIONS = 200
RANDOM_SEED = 42
## set up toolbox
toolbox = base.Toolbox()

toolbox.register('binary',random.randint,0,1)
creator.create('FitnessMax',base.Fitness,weights=(1.0,))
creator.create('Individual', list, fitness = creator.FitnessMax)
toolbox.register('IndividualCreator', tools.initRepeat, creator.Individual, toolbox.binary, chrom_size)
toolbox.register('IndividualCreator', tools.initRepeat, creator.Individual, toolbox.binary, chrom_size)
toolbox.register('PopulationCreator', tools.initRepeat, list, toolbox.IndividualCreator)

def fitnessFunction(individual):
    return sum(individual),

toolbox.register('evaluate', fitnessFunction)
toolbox.register('select',tools.selTournament, tournsize = 3)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=1/population_size)

def main():
    population = toolbox.PopulationCreator(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('max', np.max)
    stats.register('avg', np.mean)
    population, logbook = algorithms.eaSimple(population,toolbox,cxpb=P_CROSSOVER,mutpb=M_MUTATION,
                                              ngen=MAX_GENERATIONS,stats=stats,verbose=True)
    maxFitnessValues, meanFitnessValues = logbook.select('max','avg')

    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Max/Average fitness')
    plt.ylabel('Max/Average fitness over generations')
    plt.show()

if __name__ == "__main__":
    main()