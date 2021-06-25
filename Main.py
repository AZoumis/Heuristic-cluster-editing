import json
from Algorithms import agglomerative_clustering, divisive_clustering, set_k_means_i
from Graph import Grpah_random, Graph_distance, Graph_n_percent, Graph_weight, Graph_n_percent_optimized, \
    Graph_weights_optimized

graph_type = "graphs/exact"

time = 150

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

if __name__ == '__main__':
    edit_times = {}
    for ki in algorithms.keys():
        for kj in distances.keys():
            edit_times[kj + " " + ki] = {}
    for i in range(100):
        j = f'{(2 * i + 1):03d}'
        file_name = graph_type + j + ".gr"

        file = open(file_name).read()
        for ki, vi in algorithms.items():
            for kj, vj in distances.items():
                algo = kj + " " + ki
                print(file_name)
                print(algo)
                graph = from_file(file, vj)
                matrix, edits, final_time = vi(graph, time)
                print("final time:", final_time)
                print("final edit:", edits[final_time])
                edit_times[algo][file_name] = edits

    with open("output.json", "w") as outfile:
        json.dump(edit_times, outfile)
