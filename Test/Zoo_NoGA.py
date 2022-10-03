import random
from pandas import read_csv
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import numpy
import matplotlib.pyplot as plt


class Zoo:
    """This class encapsulates the Friedman1 test for a regressor
    """
    DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data'
    NUM_FOLDS = 5

    def __init__(self, randomSeed):
        """
        :param randomSeed: random seed value used for reproducible results
        """
        self.randomSeed = randomSeed
        # read the dataset, skipping the first columns (animal name):
        self.data = read_csv(self.DATASET_URL, header=None, usecols=range(1, 18))
        # separate to input features and resulting category (last column):
        self.X = self.data.iloc[:, 0:16]
        self.y = self.data.iloc[:, 16]
        # split the data, creating a group of training/validation sets to be used in the k-fold validation process:
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS)  # , random_state=self.randomSeed)
        self.classifier = DecisionTreeClassifier(random_state=self.randomSeed)

    def __len__(self):
        """
        :return: the total number of features used in this classification problem
        """
        return self.X.shape[1]

    def getMeanAccuracy(self, zeroOneList):
        """
        returns the mean accuracy measure of the calssifier, calculated using k-fold validation process,
        using the features selected by the zeroOneList
        :param zeroOneList: a list of binary values corresponding the features in the dataset. A value of '1'
        represents selecting the corresponding feature, while a value of '0' means that the feature is dropped.
        :return: the mean accuracy measure of the calssifier when using the features selected by the zeroOneList
        """
        # drop the dataset columns that correspond to the unselected features:
        zeroIndices = [i for i, n in enumerate(zeroOneList) if n == 0]
        currentX = self.X.drop(self.X.columns[zeroIndices], axis=1)
        # perform k-fold validation and determine the accuracy measure of the classifier:
        cv_results = model_selection.cross_val_score(self.classifier, currentX, self.y, cv=self.kfold,
                                                     scoring='accuracy')
        # return mean accuracy:
        return cv_results.mean()


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
                        halloffame=None, verbose=__debug__):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")
    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)
        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # add the best back to population:
        offspring.extend(halloffame.items)
        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)
        # Replace the current population by the offspring
        population[:] = offspring
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    return population, logbook


# Genetic Algorithm constants:
POPULATION_SIZE = 50
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.1  # probability for mutating an individual
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 5
FEATURE_PENALTY_FACTOR = 0.001
# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
# create the Zoo test class:
zoo = Zoo(RANDOM_SEED)

toolbox = base.Toolbox()
# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)
# create an operator that randomly returns 0 or 1:
toolbox.register("zeroOrOne", random.randint, 0, 1)
# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(zoo))
# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def zooClassificationAccuracy(individual):
    numFeaturesUsed = sum(individual)
    if numFeaturesUsed == 0:
        return 0.0,
    else:
        accuracy = zoo.getMeanAccuracy(individual)
        return accuracy - FEATURE_PENALTY_FACTOR * numFeaturesUsed,  # return a tuple


toolbox.register("evaluate", zooClassificationAccuracy)
# genetic operators:mutFlipBit
# Tournament selection with tournament size of 2:
toolbox.register("select", tools.selTournament, tournsize=2)
# Single-point crossover:
toolbox.register("mate", tools.cxOnePoint)
# Flip-bit mutation:
# indpb: Independent probability for each attribute to be flipped
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / len(zoo))


# Genetic Algorithm flow:
def main():
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)
    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    # perform the Genetic Algorithm flow with hof feature added:
    # population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
    population, logbook = eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
    # print best solution found:
    print("- Best solutions are:")
    for i in range(HALL_OF_FAME_SIZE):
        print(i, ": ", hof.items[i], ", fitness = ", hof.items[i].fitness.values[0],
              ", accuracy = ", zoo.getMeanAccuracy(hof.items[i]), ", features = ", sum(hof.items[i]))
    # extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")
    # plot statistics:

    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()


if __name__ == "__main__":
    main()
