import numpy as np
import time
import random
from queue import PriorityQueue


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
    return lambda graph, limit: k_means(graph, limit, loops)
