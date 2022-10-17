print("jsjs")

from deap import base
from deap import creator
from deap import tools
import numpy as np

DIMENSIONS = 2
POPULATION_SIZE = 20
MAX_GENERATIONS = 500
MIN_START_POSITION, MAX_START_POSITION = -5, 5
MIN_SPEED, MAX_SPEED = -3, 3
MAX_LOCAL_UPDATE_FACTOR = MAX_GLOBAL_UPDATE_FACTOR = 2.0

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", np.ndarray, fitness=creator.FitnessMin, speed=None, best=None)


def creatorParticle():
    particle = creator.Particle(np.random.uniform(MIN_START_POSITION,
                                                  MAX_START_POSITION,
                                                  DIMENSIONS))
    particle.speed = np.random.uniform(MIN_SPEED, MAX_SPEED, DIMENSIONS)
    return particle


toolbox.register("particleCreator", creatorParticle)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.particleCreator)


def updateParticle(particle, best):
    localUpdateFactor = np.random.uniform(0, MAX_LOCAL_UPDATE_FACTOR, particle.size)
    globalUpdateFactor = np.random.uniform(0, MAX_GLOBAL_UPDATE_FACTOR, particle.size)

    localSpeedUpdate = localUpdateFactor * (particle.best - particle)
    globalSpeedUpdate = globalUpdateFactor * (best - particle)

    particle.speed = particle.seed + (localSpeedUpdate + globalSpeedUpdate)

    particle[:] = particle + particle.speed


toolbox.register("update", updateParticle)


def himmelblau(particle):
    x = particle[0]
    y = particle[1]
    f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return f

toolbox.register("evaluate", himmelblau)


def main():
    print("jsjs")
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.main)
    stats.register("avg", np.mean)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    best = None
    for generation in range(MAX_GENERATIONS):
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
