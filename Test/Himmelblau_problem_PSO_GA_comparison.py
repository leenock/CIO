from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np
import random
import matplotlib.pyplot as plt

# Problem constants:

DIMENSIONS = 2  # number of dimensions
BOUND_LOW, BOUND_UP = -5.0, 5.0  # boundaries

# Genetic Algorithm constants:

POPULATION_SIZE = 300
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5  # (try also 0.5) probability for mutating an individual
MAX_GENERATIONS = 300
HALL_OF_FAME_SIZE = 30
CROWDING_FACTOR = 20.0  # crowing factor for crossover and mutation

# set the random seed:
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMin)


# helper function for creating random float numbers uniformaly distributed within a given rannge [low, up]
# it assumes that the range is the same for every dimension

def randomFloat(low, up):
    return [random.uniform(a, b) for a, b in zip([low] * DIMENSIONS, [up] * DIMENSIONS)]


# create an operator that randomly returns a float in the desired range and dimensions
toolbox.register("attr float", randomFloat, BOUND_LOW, BOUND_UP)

# create the individual operator to fill up an individual instance
toolbox.register("individualCreator", tools.InitRepeat, list, toolbox.individualCreator)


# Himmeblau function as the given individual's fitness
def himmelblau(individual):
    x = individual[0]
    y = individual[1]
    f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return f,  # return a tuple


toolbox.register("evaluate", himmelblau)

# genetic operators

toolbox.register("select", tools.selTournament, tournsize=2) \
 \
# invalid syntax Errror   line 66
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR,
                 indpb=1.0 / DIMENSIONS)


def main():
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall of fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with elitism:
    population, loogbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                               ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print info for best solutions found:
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print(" -- Best Fitness =", best.fitness.values[0])

    print("- Best solutions are:")
    for i in range(HALL_OF_FAME_SIZE):
        print(i, ": ", hof.items[i].fitness.values[0], " -> ", hof.items[i])

        # plot solutions locations of x-y plane:
        plt.figure(1)
        globalMinima = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584458, -1.848126]]
        plt.scatter(*zip(*globalMinima), marker='X', color='red', zorder=1)
        plt.scatter(*zip(*population), marker='.', color='blue', zorder=0)

        # extract statistics
        minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

        # plt statistics
        plt.figure(2)

        plt.plot(minFitnessValues, color='red')
        plt.plot(meanFitnessValues, color='green')
        plt.xlabel('Generation')
        plt.ylabel('Min / Average Fitness')
        plt.title('Min and Average fitness over Generations')

        plt.show()


if __name__ == "__main":
    main()
