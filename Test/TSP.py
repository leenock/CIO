from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import numpy as np
import matplotlib.pyplot as plt


class TSP:

    def __init__(self, towns):

        # initialize instance variables:
        self.towns = towns
        self.map = []
        self.listOfCities = []
        self.distance = 0

        # initialize the data:
        self.__initData()

    def __len__(self):
        """
        :return: the total number of towns defined in the problem
        """
        return len(self.towns)

    def __initData(self):
        self.map = np.zeros((self.towns, self.towns))
        for i in range(0, self.towns):
            for j in range(0, i):
                self.map[i, j] = random.randrange(50, 500)
                self.map[j, i] = self.map[i, j]

    ## define the fitness function

    def getFitness(self, individual):
        return sum(
            [
                self.map[individual[i], individual[i + 1]]
                for i in range(len(individual) - 1)
            ]
                  )
        map.getFitness = getFitness

        # Determine selection process

        def evaluate(self):
            distances = np.asarray(
                [self.fitness(individual) for individual in self.towns]
            )
            self.score = np.min(distances)
            self.best = self.towns[distances.tolist().index(self.score)]
            self.parents.append(self.best)
            if False in (distances[0] == distances):
                distances = np.max(distances) - distances
            return distances / np.sum(distances)

        map.evaluate = evaluate

        print(pop.evaluate())


    def printMap(self):
        return self.map


s = TSP(4)
print(s.printMap())









chrom_size = 100
population_size = 150
p_crossover = 0.9
m_mutation = 0.5
max_generations = 200
random_seed = 42

toolbox = base.Toolbox()

toolbox.register("binary", random.randint, 0, 1)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox.register("IndividualCreator", tools.initRepeat, creator.Individual, toolbox.binary, len(knapsack_prob))

toolbox.register("populationCreator", tools.initRepeat, list, toolbox.IndividualCreator)


def fitnessFuntion(individual):
    return knapsack_prob.getValue(individual),


toolbox.register("evaluate", fitnessFuntion)

toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.register("mate", tools.cxOnePoint)

toolbox.register("mutate", tools.mutFlipBit, indpb=1 / population_size)


def main():
    population = toolbox.populationCreator(n=population_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    HALL_OF_FAME_SIZE = 10
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=p_crossover, mutpb=m_mutation,
                                              ngen=max_generations,
                                              stats=stats, halloffame=hof, verbose=True)

    # population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS,stats=stats, halloffame=hof, verbose=True)
    # print best solution found:
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])
    print()
    print("-- Schedule = ")
    knapsack_prob.printItems(best)

    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('max/average fitness')
    plt.ylabel('max/average fitness over generations')
    plt.show()



if __name__ == "__main__":
    main()