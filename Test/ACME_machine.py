import numpy
import random

import numpy as np
from deap import base,tools,creator,algorithms


class ACME:
    ### parameters
    def __init__(self):
        self.totalTime = []
        self.machineTimes = []
        self.components = []
        self.machines = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        self.__config__()

    def __config__(self):
        self.machineTimes = [
            [0, 10, 15, 5, 7, 2, 8, 20, 1000],
            [10, 0, 6, 12,7,4 ,22,90,60],
            [15,6,0,33,40,900,8,10,20],
            [5,12,33,0,70,60,5,45,24],
            [7,7,40,70,0,35,20,10,11],
            [2,4,1100,60,35,0,8,1000,8],
            [8,22,8,5,20,8,0,17,55],
            [30,90,10,45,10,1000,17,0,4],
            [1000,60,20,25,11,8,55,4,0]
        ]

    def getMachineSchedule(self, totalSchedule):

    def solutionFitness(self, individual):
# 1. calc total time (add in 10 mins for every 100 mins of work
# 2. valid time (does machine time exceed the total time available

### fitness function


# Genetic Algorithm constants:
POPULATION_SIZE = 50
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.2  # probability for mutating an individual
MAX_GENERATIONS = 100
HALL_OF_FAME_SIZE = 10


# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
toolbox = base.Toolbox()
# create an operator that randomly returns 0 or 1:
toolbox.register("numbers", random.sample, range(noOfCities),noOfCities)
# define a single objective, maximizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMin)
# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.numbers)
# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# fitness calculation
def tspValue(individual):
    return tsp.calcDistance(individual),  # return a tuple

toolbox.register("evaluate", tspValue)
# genetic operators:mutFlipBit
# Tournament selection with tournament size of 3:
toolbox.register("select", tools.selTournament, tournsize=2)
# Single-point crossover:
toolbox.register("mate", tools.cxOnePoint)
# Flip-bit mutation:
# indpb: Independent probability for each attribute to be flipped
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/noOfCities)

# Genetic Algorithm flow:
def main():
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=False)
    # print best solution found:
    best = hof.items[0]
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness/distance = ", best.fitness.values[0])
    print("city sequence: ", tsp.printCities(best))
    #print("-- Knapsack Items = ")
    #knapsack.printItems(best)
    # extract statistics:
    #maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")
    # plot statistics:

if __name__ == "__main__":
    main()







