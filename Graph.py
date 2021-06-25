import numpy as np
from queue import PriorityQueue

# Contains implementations of graph with different distance metrics.

class Vertex:
    def __init__(self):
        self.connections = set()

    def add_edge(self, neighbour):
        self.connections.add(neighbour)

    def remove_edge(self, neighbour):
        self.connections.remove(neighbour)

    def get_neighbours(self):
        return self.connections


class Graph:
    def __init__(self, n, m):
        self.vertices = []
        self.n = n
        self.m = m
        self.n_matrix = np.zeros((n, n))
        self.distances = {}
        for i in range(n):
            self.vertices.append(Vertex())
            self.vertices[i].add_edge(i)
            self.n_matrix[i][i] = 1

    def add_edge(self, a, b):
        self.vertices[a].add_edge(b)
        self.vertices[b].add_edge(a)
        self.n_matrix[a][b] = 1
        self.n_matrix[b][a] = 1

    def get_vertices(self):
        return self.vertices

    def get_graph_matrix(self):
        return self.n_matrix


class Graph_n_percent(Graph):
    def distance(self, vs):
        """A larger value indicates that the clusters are closer"""
        if vs not in self.distances:
            intersection = set.intersection(*list(map(lambda x: x.get_neighbours(), [self.vertices[i] for i in vs])))
            union = set.union(*list(map(lambda x: x.get_neighbours(), [self.vertices[i] for i in vs])))
            self.distances[vs] = len(intersection) / len(union)
        return self.distances[vs]

    def midpoint(self):
        return 0.5

    def max_value(self):
        return 1

    def min_value(self):
        return 0


class Graph_n_percent_optimized(Graph):
    def distance(self, vs):
        """A larger value indicates that the clusters are closer"""
        if vs not in self.distances:
            intersection = set.intersection(*list(map(lambda x: x.get_neighbours(), [self.vertices[i] for i in vs])))
            if len(intersection) <= 1:
                self.distances[vs] = 0
            else:
                union = set.union(*list(map(lambda x: x.get_neighbours(), [self.vertices[i] for i in vs])))
                self.distances[vs] = len(intersection) / len(union)

        return self.distances[vs]

    def midpoint(self):
        return 0.5

    def max_value(self):
        return 1

    def min_value(self):
        return 0


class Graph_weight(Graph):
    def distance(self, vs):
        """A larger value indicates that the clusters are closer"""
        if vs not in self.distances:
            intersection = set.intersection(*list(map(lambda x: x.get_neighbours(), [self.vertices[i] for i in vs])))
            union = set.union(*list(map(lambda x: x.get_neighbours(), [self.vertices[i] for i in vs])))
            self.distances[vs] = len(intersection) - (len(union) / 2)
        return self.distances[vs]

    def midpoint(self):
        return 0

    def max_value(self):
        return float('inf')

    def min_value(self):
        return float('-inf')


class Graph_weights_optimized(Graph):
    def distance(self, vs):
        """A larger value indicates that the clusters are closer"""
        if vs not in self.distances:
            intersection = set.intersection(*list(map(lambda x: x.get_neighbours(), [self.vertices[i] for i in vs])))
            if len(intersection) <= 1:
                self.distances[vs] = float('-inf')
            else:
                union = set.union(*list(map(lambda x: x.get_neighbours(), [self.vertices[i] for i in vs])))
                if len(intersection) == len(union):
                    self.distances[vs] = float('inf')
                else:
                    self.distances[vs] = len(intersection) - (len(union) / 2)
        return self.distances[vs]

    def midpoint(self):
        return 0

    def max_value(self):
        return float('inf')

    def min_value(self):
        return float('-inf')


class Graph_distance(Graph):
    def distance(self, vs):
        """dijkstra, Finds the max distance in vs"""
        if vs not in self.distances:
            self.distances[vs] = float('inf')
            vs_list = list(vs)
            for i in range(len(vs_list)):
                for j in range(i + 1, len(vs_list)):
                    a = vs_list[i]
                    b = vs_list[j]
                    ab = frozenset([a, b])
                    if ab not in self.distances:
                        self.distances[ab] = float('-inf')
                        q = PriorityQueue()
                        found = False
                        visited = set([a])
                        q.put((0, a))
                        while not found and not q.empty():
                            dist, curr = a.pop(a)
                            for i in curr.get_neighbours():
                                if i == b:
                                    self.distances[vs] = -(dist + 1)
                                    found = True
                                    break
                                if i not in visited:
                                    visited.add(i)
                                    q.put((dist + 1, i))

                    if self.distances[vs] > self.distances[ab]:
                        self.distances[vs] = self.distances[ab]

        return self.distances[vs]

    def midpoint(self):
        return 2

    def max_value(self):
        return 0

    def min_value(self):
        return float('-inf')


class Grpah_random(Graph):
    def distance(self, vs):
        """A larger value indicates that the clusters are closer.
        Not actually random, but based on hash"""
        if vs not in self.distances:
            self.distances[vs] = hash(vs)
        return self.distances[vs]

    def midpoint(self):
        return self.n

    def max_value(self):
        return float('inf')

    def min_value(self):
        return float('-inf')

