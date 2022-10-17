from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np
import random
import numpy
import matplotlib.pyplot as plt

def main():

    population = toolbox.populationCreator(n=POPULATION_SIZE)

    stats = tools.Statistics(lambda  ind: ind.fitness.values)
    stats.register("min", np.main)
    stats.register("avg", np.mean)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"]+ stats.fields

    best = None

    for generation in range(MAX_GENERATION)

        for particle in population:
            particle.fitness.values = toolbox.evaluate(particle)

            if particle.best is None or particle.best.size == 0 or particle.best.fitness < particle.fitness:
                particle.best = creator.Particle(particle)
                particle.best.fitness.values = particle.fitness.values

            if best is None or best.size == 0 or best.fitness < particle.fitness:
                best = creator.Particle(particle)
                best.fitness.values = particle.fitness.values

        for particle in population:
            toolbox.update(particle, best)

        logbook.record(gen=generation, evals=len(population), **stats.compile(population))
        print(logbook.stream)

    print("-- Best Particle = ", best)
    print("-- Best Fitness =", best.fitness.values[0])

if __import__ == "__main__":
    main()
