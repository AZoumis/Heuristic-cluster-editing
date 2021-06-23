# Developing efficient heuristic approaches for cluster editing, inspired by other clustering problems
### Angelos Zoumis

This repository contains the algorithms used during the experiments for the research project of Developing efficient heuristic approaches for cluster editing, inspired by other clustering problems by Angelos Zoumis.

## Running the algorithm
In order to run the code, python 3 and numpy is required. Furthermore, the graphs can be downloaded from here: https://pacechallenge.org/2021/tracks

To change the source of the files, line 450 can be changed.

From line 422 to 444, the combination of algorithms and distance metrics that will be used are declared. The code will produce the file output.json, which will contain be in the following format:

{"\<algorithm\> \<distance metric\>": {"\<graph file name\>": {\<time\>: \<edits\>, ...}, ...}, ...}

## Experiment output
The json file test-1.json contains the results of the test with the 6 distance metrics combined with the 3 different algorithms. test-2.json contains the results from the experiments using different i for the k-means algorithm.
