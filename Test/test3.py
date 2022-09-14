from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class knapsack:
    def __init__(self, penalty_val=30):
        self.penalty_val = penalty_val
        self.weights = [9, 13, 153, 50, 15, 68, 27, 39, 23, 52, 11, 32, 24, 48, 73, 42, 43, 22, 7, 18, 4, 30]
        self.value = [150, 35, 200, 160, 60, 45, 60, 40, 30, 10, 70, 30, 15, 10, 40, 70, 75, 80, 20, 12, 50, 10]
        self.ItemName = np.array(
            ['map', 'compass', 'water', 'sandwich', 'glucose', 'tin', 'banana', 'apple', 'cheese', 'beer',
             'santan cream'
                , 'camera', 'T-shirt', 'trousers', 'umbrella', 'waterproof trousers', 'waterproof overclothes',
             'note-case'
                , 'sunglasses', 'towel', 'socks', 'book'])

    def __len__(self):
        return len(self.weights)

    def getCost(self, ItemsList):
        self.ItemsList = np.array(ItemsList)

        WeightCost = 0 if (400 - self.getWeightCost(ItemsList)) > 0 else (400 - self.getWeightCost(ItemsList))
        ValueCost = self.getValueCost(ItemsList)

        return WeightCost * self.penalty_val + ValueCost

    def getWeightCost(self, ItemsList):
        return sum([ItemsList[x] * self.weights[x] for x in range(len(ItemsList))])

    def getValueCost(self, ItemsList):
        return sum([ItemsList[x] * self.value[x] for x in range(len(ItemsList))])

    def printItemAndCost(self, ItemsList):
        print("List of the items: ")

        for x in np.where(self.ItemsList == 1):
            print(self.ItemName[x])

        print("Weight Cost: ", self.getWeightCost(ItemsList))
        print("Value Gain: ", self.getValueCost(ItemsList))


POPULATION_SIZE = 300
P_CROSSOVER = 0.8
P_MUTATION = 0.3
MAX_GENERATIONS = 200
random.seed(42)
toolbox = base.Toolbox()
knapsack_prob = knapsack()
# define a single objective, maximizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMin)
# create an operator that randomly returns 0 or 1:
toolbox.register("binary", random.randint, 0, 1)
# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.binary, len(knapsack_prob))
# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# define the fitness function / objective function
def GenerateKnapsac(individual):
    return knapsack_prob.getCost(individual),  # return a tuple


toolbox.register("evaluate", GenerateKnapsac)
# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=4)
# toolbox.register("select", tools.selRoulette)
# toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / len(knapsack_prob))
# create initial population (generation 0):
population = toolbox.populationCreator(n=POPULATION_SIZE)
# prepare the statistics object:
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.min)
stats.register("avg", np.mean)
# define the hall-of-fame object:
HALL_OF_FAME_SIZE = 10
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS,
                                          stats=stats, halloffame=hof, verbose=True)
# population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS,stats=stats, halloffame=hof, verbose=True)
# print best solution found:
best = hof.items[0]
print("-- Best Individual = ", best)
print("-- Best Fitness = ", best.fitness.values[0])
print()
print("-- Schedule = ")
knapsack_prob.printItemAndCost(best)
# extract statistics:
minFitnessValues, meanFitnessValues = logbook.select("max", "avg")
# plot statistics:
plt.plot(minFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Max / Average Fitness')
plt.title('Max and Average fitness over')