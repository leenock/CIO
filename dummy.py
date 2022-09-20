import numpy as np


class Population:
    def __init__(self, bag, adjacency_mat):
        self.bag = bag  # to represent the full population,
        self.parents = []  # to represent th chosen, selected superior few,
        self.score = 0  # to store the score of the best chromosome in the population
        self.best = None  # best to store the best chromosome itself
        self.adjacency_mat = adjacency_mat  # the adjacency matrix that we will be using to calculate the distance in # the context of TSP.

    # Here is a little snippet of code that we will be using to randomly generate the first generation of population.

    def init_population(cities, adjacency_mat, n_population):
        return Population(
            np.asarray([np.random.permutation(cities) for _ in range(n_population)]),
            adjacency_mat
        )

    # Let’s see if this everything works as expected by generating a dummy population.

        pop = init_population(cities, adjacency_mat, 5)
        pop.bag

   # some function that will determine the fitness of a chromosome.

    def fitness(self, chromosome):
        return sum(
            [
                self.adjacency_mat[chromosome[i], chromosome[i + 1]]
                for i in range(len(chromosome) - 1)
            ]
        )


        Population.fitness = fitness

      #  we evaluate the population

        def evaluate(self):
            distances = np.asarray(
                [self.fitness(chromosome) for chromosome in self.bag]
            )
            self.score = np.min(distances)
            self.best = self.bag[distances.tolist().index(self.score)]
            self.parents.append(self.best)
            if False in (distances[0] == distances):
                distances = np.max(distances) - distances
            return distances / np.sum(distances)

        Population.evaluate = evaluate


        #
        pop.evaluate()
        pop.best
        pop.score

        def select(self, k=4):
            fit = self.evaluate()
            while len(self.parents) < k:
                idx = np.random.randint(0, len(fit))
                if fit[idx] > np.random.rand():
                    self.parents.append(self.bag[idx])
            self.parents = np.asarray(self.parents)

        Population.select = select

        def swap(chromosome):
            a, b = np.random.choice(len(chromosome), 2)
            chromosome[a], chromosome[b] = (
                chromosome[b],
                chromosome[a],
            )
            return chromosome

        def crossover(self, p_cross=0.1):
            children = []
            count, size = self.parents.shape
            for _ in range(len(self.bag)):
                if np.random.rand() > p_cross:
                    children.append(
                        list(self.parents[np.random.randint(count, size=1)[0]])
                    )
                else:
                    parent1, parent2 = self.parents[
                                       np.random.randint(count, size=2), :
                                       ]
                    idx = np.random.choice(range(size), size=2, replace=False)
                    start, end = min(idx), max(idx)
                    child = [None] * size
                    for i in range(start, end + 1, 1):
                        child[i] = parent1[i]
                    pointer = 0
                    for i in range(size):
                        if child[i] is None:
                            while parent2[pointer] in child:
                                pointer += 1
                            child[i] = parent2[pointer]
                    children.append(child)
            return children

        Population.crossover = crossover

        def mutate(self, p_cross=0.1, p_mut=0.1):
            next_bag = []
            children = self.crossover(p_cross)
            for child in children:
                if np.random.rand() < p_mut:
                    next_bag.append(swap(child))
                else:
                    next_bag.append(child)
            return next_bag

        Population.mutate = mutate

        def genetic_algorithm(
                cities,
                adjacency_mat,
                n_population=5,
                n_iter=20,
                selectivity=0.15,
                p_cross=0.5,
                p_mut=0.1,
                print_interval=100,
                return_history=False,
                verbose=False,
        ):
            pop = cities.init_population(cities, adjacency_mat, n_population)
            best = pop.best
            score = float("inf")
            history = []
            for i in range(n_iter):
                pop.select(n_population * selectivity)
                history.append(pop.score)
                if verbose:
                    print(f"Generation {i}: {pop.score}")
                elif i % print_interval == 0:
                    print(f"Generation {i}: {pop.score}")
                if pop.score < score:
                    best = pop.best
                    score = pop.score
                children = pop.mutate(p_cross, p_mut)
                pop = Population(children, pop.adjacency_mat)
            if return_history:
                return best, history
            return best

        genetic_algorithm(cities, adjacency_mat, verbose=True)

        best, history = genetic_algorithm(
            cities,
            adjacency_mat,
            n_iter=100,
            verbose=False,
            print_interval=20,
            return_history=True,
        )

        plt.plot(range(len(history)), history, color="skyblue")
        plt.show()
        print(best)



