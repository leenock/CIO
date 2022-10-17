import numpy as np

from deap import base
from deap import creator
from deap import tools






# Himmelblau function:
def himmelblau(particle):
    x = particle[0]
    y = particle[1]
    f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return f,  # return a tuple


toolbox.register("evaluate", himmelblau)

def main():
    # create the population of particle population:
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields


best = None
    for generation in range(MAX_GENERATIONS):

        # evaluate all particles in polulation:
        for particle in population:

            # find the fitness of the particle:
            particle.fitness.values = toolbox.evaluate(particle)

            # particle best needs to be updated:
            if particle.best is None or particle.best.size == 0 or particle.best.fitness < particle.fitness:
                particle.best = creator.Particle(particle)
                particle.best.fitness.values = particle.fitness.values

            # global best needs to be updated:
            if best is None or best.size == 0 or best.fitness < particle.fitness:
                best = creator.Particle(particle)
                best.fitness.values = particle.fitness.values

        # update each particle's speed and position:
        for particle in population:
            toolbox.update(particle, best)

        # record the statistics for the current generation and print it:
        logbook.record(gen=generation, evals=len(population), **stats.compile(population))
        print(logbook.stream)

    # print info for best solution found:
    print("-- Best Particle = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])


if __name__ == "__main__":
    main()