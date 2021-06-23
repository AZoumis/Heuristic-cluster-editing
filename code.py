import numpy as np
# import matplotlib.pyplot as plt
import time
import statistics
import random
import json
from queue import PriorityQueue

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

def edit_diff(a, b):
    return np.sum(np.abs(a - b))


def agglomerative_clustering(graph, limit):
    start = time.process_time()

    clusters = set(map(lambda x: frozenset([x]), range(graph.n)))
    last_matrix = np.zeros((graph.n, graph.n))
    for i in range(graph.n):
        last_matrix[i][i] = 1
    best_matrix = last_matrix
    best_edit = edit_diff(last_matrix, graph.get_graph_matrix())
    edits = {}

    while (time.process_time() - start) < limit:
        final_time = time.process_time() - start
        edits[final_time] = best_edit / 2
        max_distance = graph.min_value()
        max_clusters = None
        for i in clusters:
            for j in clusters:
                if i != j:
                    distance = graph.distance(i.union(j))
                    if distance > max_distance:
                        max_distance = distance
                        max_clusters = i, j
        if max_clusters is None:
            return best_matrix, edits, final_time

        else:
            last_matrix = np.zeros((graph.n, graph.n))
            clusters.remove(max_clusters[0])
            clusters.remove(max_clusters[1])
            clusters.add(frozenset(max_clusters[0].union(max_clusters[1])))
            for cluster in clusters:
                for i in cluster:
                    for j in cluster:
                        last_matrix[i][j] = 1
            edit = edit_diff(last_matrix, graph.get_graph_matrix())
            if edit < best_edit:
                best_edit = edit
                best_matrix = last_matrix

    final_time = time.process_time() - start
    edits[final_time] = best_edit / 2
    return best_matrix, edits, final_time

def divisive_clustering(graph, limit):
    start = time.process_time()

    cluster = frozenset(range(graph.n))
    clusters = set([cluster])
    for i in graph.get_vertices():
        i.cluster = cluster
    last_matrix = np.ones((graph.n, graph.n))
    best_matrix = last_matrix
    best_edit = edit_diff(last_matrix, graph.get_graph_matrix())
    edits = {}
    q = PriorityQueue()
    for i in range(graph.n):
        for j in range(i + 1, graph.n):
            q.put((graph.distance(frozenset([i, j])), (i, j)))

    while (time.process_time() - start) < limit and not q.empty():
        final_time = time.process_time() - start
        edits[final_time] = best_edit / 2
        split = q.get()
        l = split[1][0]
        r = split[1][1]
        if graph.get_vertices()[l].cluster == graph.get_vertices()[r].cluster:
            cluster = graph.get_vertices()[l].cluster
            l_cluster = set([l])
            graph.get_vertices()[l].cluster = l_cluster
            r_cluster = set([r])
            graph.get_vertices()[r].cluster = r_cluster
            for i in cluster:
                if i == l or i == r:
                    continue
                # elif graph.distance(frozenset(l_cluster.union(set([i])))) > graph.distance(frozenset(r_cluster.union(set([i])))):
                # elif graph.distance(frozenset([l, i])) > graph.distance(frozenset([r, i])) and graph.distance(frozenset([l, i])) >= graph.midpoint():
                elif graph.distance(frozenset([l, i])) > graph.distance(frozenset([r, i])):
                    l_cluster.add(i)
                    graph.get_vertices()[i].cluster = l_cluster
                else:
                    r_cluster.add(i)
                    graph.get_vertices()[i].cluster = r_cluster
            clusters.remove(cluster)
            clusters.add(frozenset(l_cluster))
            clusters.add(frozenset(r_cluster))

            last_matrix = np.zeros((graph.n, graph.n))
            for cluster in clusters:
                for i in cluster:
                    for j in cluster:
                        last_matrix[i][j] = 1
            edit = edit_diff(last_matrix, graph.get_graph_matrix())
            if edit < best_edit:
                best_edit = edit
                best_matrix = last_matrix

    final_time = time.process_time() - start
    edits[final_time] = best_edit / 2
    return best_matrix, edits, final_time


def k_means(graph, limit, loops):
    random.seed(42)
    n_list = range(graph.n)
    start = time.process_time()
    best_edit = graph.m
    best_clusters = map(lambda x: set([x]), range(graph.n))
    edits = {}
    k = int((graph.n * (graph.n + 1) / 2) / graph.m)

    if k == graph.n:
        k += -1
    delta_k = 0
    while (time.process_time() - start) < limit:
        k += delta_k
        if k > graph.n or k < 1:
            delta_k = (delta_k) * (-1)
            delta_k += -1 if delta_k < 0 else 1
            k += delta_k
            if k > graph.n or k < 1:
                break
        delta_k = (delta_k) * (-1)
        delta_k += -1 if delta_k < 0 else 1

        prev_clusters = np.array([random.sample(n_list, k)], dtype=int).T

        for itt in range(loops):
            final_time = time.process_time() - start
            edits[final_time] = best_edit
            clusters = list(map(lambda x: [], range(len(prev_clusters))))
            values = []
            for i in range(len(prev_clusters)):
                if prev_clusters[i]:
                    rand = random.choice(prev_clusters[i])
                    values.append(rand)
                    graph.get_vertices()[rand].cluster = i
            for i in n_list:
                closest = None
                closest_dist = graph.min_value()
                picked = None
                for j in range(len(values)):
                    if values[j] == -1:
                        continue
                    if i == values[j]:
                        picked = j
                        continue
                    dist = graph.distance(frozenset((i, values[j])))
                    if dist > closest_dist:
                        closest = j
                        closest_dist = dist

                if picked is not None:
                    if closest_dist == graph.max_value():
                        if picked > closest:
                            values[picked] = -1
                            clusters[closest] = clusters[picked]
                        else:
                            values[closest] = -1
                            clusters[picked] = clusters[closest]
                    else:
                        closest = picked
                if closest is None:
                    closest = len(values)
                    clusters.append([])
                    values.append(i)

                clusters[closest].append(i)
                graph.get_vertices()[i].cluster = closest

            last_n_edits = 0
            last_clusters = np.full(graph.n, set())
            for i in range(graph.n):
                last_clusters[i] = set(clusters[graph.get_vertices()[i].cluster])
                last_n_edits += len(graph.get_vertices()[i].get_neighbours().union(last_clusters[i])) - len(
                    graph.get_vertices()[i].get_neighbours().intersection(last_clusters[i]))

            last_n_edits = last_n_edits / 2
            if last_n_edits < best_edit:
                best_clusters = np.copy(last_clusters)
                best_edit = last_n_edits
            prev_clusters = clusters

    best_matrix = np.zeros((graph.n, graph.n))
    for cluster in best_clusters:
        for i in cluster:
            for j in cluster:
                best_matrix[i][j] = 1

    edits[final_time] = best_edit
    return best_matrix, edits, final_time


def set_k_means_i(loops):
    return (lambda graph, limit: k_means(graph, limit, loops))

def set_k_means_i(loops):
    return (lambda graph, limit: k_means(graph, limit, loops))

def from_file(file, graph_type):
    lines = file.split("\n")
    graph = None
    for i in lines:
        line = i.split()
        if not line or line[0] == "c":
            continue
        if line[0] == "p":
            n = int(line[2])
            m = int(line[3])
            graph = graph_type(n, m)
        else:
            a = int(line[0])
            b = int(line[1])
            graph.add_edge(a - 1, b - 1)
    return graph

def to_file(original, final):
    res = ""
    for i in range(len(original)):
        for j in range(len(original)):
            if i < j and original[i][j] != final[i][j]:
                res += str(i + 1) + " " + str(j + 1) + "\n"
    return res

algorithms = {
    "agglomerative clustering": agglomerative_clustering,
    "divisive clustering": divisive_clustering,
    "k-means i = 1": set_k_means_i(1),
    "k-means i = 2": set_k_means_i(2),
    "k-means i = 4": set_k_means_i(4),
    "k-means i = 8": set_k_means_i(8),
    "k-means i = 16": set_k_means_i(16),
    "k-means i = 32": set_k_means_i(32),
    "k-means i = 64": set_k_means_i(64),
    "k-means i = 128": set_k_means_i(128),
    "k-means i = 256": set_k_means_i(256),
    "k-means i = 512": set_k_means_i(512),
    }
distances = {
    "random": Grpah_random,
    "dijkstra": Graph_distance,
    "neighbourhood%": Graph_n_percent,
    "neighbourhood difference": Graph_weight,
    "neighbourhood% optimized": Graph_n_percent_optimized,
    "neighbourhood difference optimized": Graph_weights_optimized
}

edit_times = {}
graph_type = "exact"
for ki in algorithms.keys():
    for kj in distances.keys():
        edit_times[kj + " " + ki] = {}
for i in range(100):
    j = f'{(2 * i + 1):03d}'
    file_name = "../" + graph_type + "/" + graph_type + j + ".gr"

    file = open(file_name).read()
    for ki, vi in algorithms.items():
        for kj, vj in distances.items():
            algo = kj + " " + ki
            print(file_name)
            print(algo)
            graph = from_file(file, vj)
            matrix, edits, final_time = vi(graph, 150)
            print("final time:", final_time)
            print("final edit:", edits[final_time])
            edit_times[algo][file_name] = edits

with open("output.json", "w") as outfile:
    json.dump(edit_times, outfile)
