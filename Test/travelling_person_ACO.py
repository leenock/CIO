import numpy as np
import random
import matplotlib.pyplot as plt
import random


class Graph:
    def __init__(self, n):

        self.amount_vertices = n
        self.map = np.zeros((n, n))
        self.trail = np.zeros((n, n))

        for i in range(0, n):
            for j in range(0, i):
                self.map[i, j] = random.randrange(100, 500)
                self.map[j, i] = self.map[i, j]

        for i in range(0, n):

            for j in range(0, i):
                self.trail[i, j] = 0.1
                self.trail[j, i] = self.trail[i, j]

    def getCostPath(self, path):

        fit = 0
        pair_list = list(zip(path, path[1:]))
        for i in pair_list:
            dist = i
            fit = fit + self.map[dist]
        return fit

    def calcCityScore(self, currentCity, nextcity, alpha, beta):

        distance = self.map[currentCity][nextcity]

        pheromone = self.trail[currentCity][nextcity]

        candidateCityScore = pheromone * alpha * (1 / distance) * beta

        return candidateCityScore


class ant:
    def __init__(self, graph):
        self.graph = graph
        self.map = graph.map
        self.trail = graph.trail

        self.city_distance = []

    def get_trail(self):
        scope = 1
        current = 0

        cities_score = {}
        total_score = 0
        cities_prob = {}
        route = []
        while scope < (self.graph.amount_vertices + 1):
            for i in range(self.graph.amount_vertices):
                # generate the score of each city for this round
                if i == current:
                    score = 0
                else:
                    score = self.graph.calcCityScore(current, i, 2, 4)
                total_score += score
                cities_score[i] = score

            for i, v in cities_score.items():
                prob = v / total_score

                cities_prob[i] = prob

            next_city = self.weighted_random_choice(cities_prob)

            while next_city in route:
                del cities_prob[next_city]

                next_city = self.weighted_random_choice(cities_prob)
            route.append(next_city)
            current = next_city

            scope += 1

        route.remove(0)
        route.append(0)

        city_distance = []
        for i in range(len(route) - 1):
            dist = self.graph.map[i][i + 1]
            city_distance.append(dist)

        return route

    def update_trail(self, route):
        for i in range(len(route) - 1):
            x = route[i]
            y = route[i + 1]

            dist = self.graph.map[x][y]
            self.city_distance.append(dist)

        for i in range(len(route) - 1):
            self.graph.trail[route[i]][route[i + 1]] = self.graph.trail[route[i]][route[i + 1]] + 1 / \
                                                       self.city_distance[i]
            self.graph.trail[route[i + 1]][route[i]] = self.trail[route[i]][route[i + 1]]
        return dist

    def weighted_random_choice(selfself, choices):
        max = sum(choices.values())
        pick = random.uniform(0, max)
        current = 0
        for key, value in choices.items():
            current += value
            if current > pick:
                return key


class colony:
    def __init__(self, graph, population, iterations, alpha, beta):
        self.graph = graph
        self.population = population
        self.iteration = iterations
        self.alpha = alpha
        self.beta = beta

        self.colony = []
        self.best_route = []
        self.best_distance = 3000

        for i in range(0, population):
            new_ant = ant(graph)
            self.colony.append(new_ant)

    def run(self):
        for i in range(self.iteration):
            for ant in self.colony:
                Temproute = ant.get_trail()
                route = [0] + Temproute
                cost = self.graph.getCostPath(route)

                ant.update_trail(route)

                if self.best_distance > cost:
                    self.best_distance = cost
                    self.best_route = route

                    # depreciate the pheromone trail

        print("Best route and cost: ", self.best_route, self.best_distance)


random.seed(42)
graph = Graph(15)
print(graph.map)
new_colony = colony(graph, 100, 100, 0.4, 0.7)
new_colony.run()
