from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import numpy

import matplotlib.pyplot as plt


# In[2]:
class Guard:

    def __init__(self, hardConstraintPenalty):

        self.hardConstraintPenalty = hardConstraintPenalty

        # list of guards:
        self.guards = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        self.shiftPreference = [[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1]]

        self.shiftMin = [2, 2, 1]
        self.shiftMax = [3, 4, 2]

        # max shifts per week allowed for each guard
        self.maxShiftsPerWeek = 5
        # number of weeks we create a schedule for:
        self.weeks = 1
        # useful values:
        self.shiftPerDay = len(self.shiftMin)
        self.shiftsPerWeek = 7 * self.shiftPerDay

    def __len__(self):

        return len(self.guards) * self.shiftsPerWeek * self.weeks

    def getCost(self, schedule):

        guardShiftsDict = self.getGuardShifts(schedule)

        consecutiveShiftViolations = self.countConsecutiveShiftViolations(guardShiftsDict)
        shiftsPerWeekViolations = self.countShiftsPerWeekViolations(guardShiftsDict)[1]
        guardsPerShiftViolations = self.countGuardsPerShiftViolations(guardShiftsDict)[1]
        shiftPreferenceViolations = self.countShiftPreferenceViolations(guardShiftsDict)

        hardContstraintViolations = consecutiveShiftViolations + guardsPerShiftViolations + shiftsPerWeekViolations
        softContstraintViolations = shiftPreferenceViolations

        return self.hardConstraintPenalty * hardContstraintViolations + softContstraintViolations

    def getGuardShifts(self, schedule):

        shiftsPerGuard = self.__len__() // len(self.guards)
        guardShiftsDict = {}
        shiftIndex = 0
        for guard in self.guards:
            guardShiftsDict[guard] = schedule[shiftIndex:shiftIndex + shiftsPerGuard]
            shiftIndex += shiftsPerGuard

        return guardShiftsDict

    def countConsecutiveShiftViolations(self, guardShiftsDict):

        violations = 0
        # iterate over the shifts of each guard:
        for guardShifts in guardShiftsDict.values():
            # look for two cosecutive '1's:
            for shift1, shift2 in zip(guardShifts, guardShifts[1:]):
                if shift1 == 1 and shift2 == 1:
                    violations += 1
        return violations

    def countShiftsPerWeekViolations(self, guardShiftsDict):

        violations = 0
        weeklyShiftsList = []
        # iterate over the shifts of each guard:
        for guardShifts in guardShiftsDict.values():  # all shifts of a single guard
            # iterate over the shifts of each weeks:
            for i in range(0, self.weeks * self.shiftsPerWeek, self.shiftsPerWeek):
                # count all the '1's over the week:
                weeklyShifts = sum(guardShifts[i:i + self.shiftsPerWeek])
                weeklyShiftsList.append(weeklyShifts)
                if weeklyShifts > self.maxShiftsPerWeek:
                    violations += weeklyShifts - self.maxShiftsPerWeek

        return weeklyShiftsList, violations

    def countGuardsPerShiftViolations(self, guardShiftsDict):

        totalPerShiftList = [sum(shift) for shift in zip(*guardShiftsDict.values())]

        violations = 0
        # iterate over all shifts and count violations:
        for shiftIndex, numOfGuards in enumerate(totalPerShiftList):
            dailyShiftIndex = shiftIndex % self.shiftPerDay  # -> 0, 1, or 2 for the 3 shifts per day
            if (numOfGuards > self.shiftMax[dailyShiftIndex]):
                violations += numOfGuards - self.shiftMax[dailyShiftIndex]
            elif (numOfGuards < self.shiftMin[dailyShiftIndex]):
                violations += self.shiftMin[dailyShiftIndex] - numOfGuards

        return totalPerShiftList, violations

    def countShiftPreferenceViolations(self, guardShiftsDict):

        violations = 0
        for guardIndex, shiftPreference in enumerate(self.shiftPreference):
            # duplicate the shift-preference over the days of the period
            preference = shiftPreference * (self.shiftsPerWeek // self.shiftPerDay)
            # iterate over the shifts and compare to preferences:
            shifts = guardShiftsDict[self.guards[guardIndex]]
            for pref, shift in zip(preference, shifts):
                if pref == 0 and shift == 1:
                    violations += 1
        return violations

    def printScheduleInfo(self, schedule):

        guardShiftsDict = self.getGuardShifts(schedule)

        print("Schedule for each guard:")
        for guard in guardShiftsDict:  # all shifts of a single guard
            print(guard, ":", guardShiftsDict[guard])

        print("consecutive shift violations = ", self.countConsecutiveShiftViolations(guardShiftsDict))
        print()

        weeklyShiftsList, violations = self.countShiftsPerWeekViolations(guardShiftsDict)
        print("weekly Shifts = ", weeklyShiftsList)
        print("Shifts Per Week Violations = ", violations)
        print()

        totalPerShiftList, violations = self.countGuardsPerShiftViolations(guardShiftsDict)
        print("Guards Per Shift = ", totalPerShiftList)
        print("Guards Per Shift Violations = ", violations)
        print()

        shiftPreferenceViolations = self.countShiftPreferenceViolations(guardShiftsDict)
        print("Shift Preference Violations = ", shiftPreferenceViolations)
        print()


HARD_CONSTRAINT_PENALTY = 10
POPULATION_SIZE = 300
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 100
random.seed(42)

toolbox = base.Toolbox()

# create the guard scheduling problem instance to be used:
gsp = Guard(HARD_CONSTRAINT_PENALTY)

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMin)

# create an operator that randomly returns 0 or 1:
toolbox.register("binary", random.randint, 0, 1)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.binary, len(gsp))

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def getCost(individual):
    return gsp.getCost(individual),


toolbox.register("evaluate", getCost)

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("select", tools.selRoulette)
# toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mate", tools.cxOnePoint)

toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / len(gsp))

# create initial population (generation 0):
population = toolbox.populationCreator(n=POPULATION_SIZE)

# prepare the statistics object:
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", numpy.min)
stats.register("avg", numpy.mean)

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
gsp.x(best)

# extract statistics:
minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

# plot statistics:
plt.plot(minFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Min / Average Fitness')
plt.title('Min and Average fitness over Generations')
plt.show()
