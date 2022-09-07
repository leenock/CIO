from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import numpy as np

import matplotlib.pyplot as plt
import Guards as gd

chrom_size = 100
population_size = 200
p_crossover = 0.9
m_mutation = 0.1
max_generations = 200
random_seed = 42

toolbox = base.Toolbox()

toolbox.register("binary", random.randint, 0, 1)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox.register("IndividualCreator", tools.initRepeat, creator.Individual, toolbox.binary, chrom_size)

toolbox.register("populationCreator", tools.initRepeat, list, toolbox.IndividualCreator)


def fitnessFuntion(individual):
    return sum(individual),


toolbox.register("evaluate", fitnessFuntion)

toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.register("mate", tools.cxTwoPoint)

toolbox.register("mutate", tools.mutFlipBit, indpb=1 / population_size)


def main():
    population = toolbox.populationCreator(n=population_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=p_crossover,
                                              mutpb=m_mutation, ngen=max_generations,
                                              stats=stats, verbose=True)
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('max/average fitness')
    plt.ylabel('max/average fitness over generations')
    plt.show()


if __name__ == "__main__":
    main()
