import numpy
import random

import numpy as np
from deap import base,tools,creator,algorithms
# encode info into TSP class
class TSP:
    #number of cities
    # map to represent distances between cities
    # a value to represent total distance covered
    # list of city names
    def __init__(self, cityNo):
        self.cityNo = cityNo
        self.map = []
        self.listOfCities = ["KL","singapore","bangkok","penang","JB","langkawi","London","New york"]
        self.distance = 0
        self.__initMap()
    def __initMap(self):
        self.map = np.zeros((self.cityNo,self.cityNo))
        for i in range(0,self.cityNo):
            for j in range(0, i):
                self.map[i][j] = random.randrange(10,100)
                self.map[j][i] = self.map[i][j]
    # a function to evaluate solutions - fitness function
    def calcDistance(self, individual):  # [1,3,4,2]
        pair_cities = list(zip(individual, individual[1:]))#[(1,3),(3,4),(4,2)]
        for i in pair_cities:#[(1,3),(3,4),(4,2)]
            self.distance = self.distance + self.map[i]
        return self.distance
    def printCities(self,individual):
        cities = []
        for i in individual:
            cities.append(self.listOfCities[i])
        return cities
noOfCities = 8
tsp = TSP(noOfCities)
##### set up the GA

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