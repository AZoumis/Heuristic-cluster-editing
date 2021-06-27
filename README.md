# Developing efficient heuristic approaches to cluster editing, inspired by other clustering problems
### Angelos Zoumis

This repository contains the algorithms used during the experiments for the research project of Developing efficient heuristic approaches to cluster editing, inspired by other clustering problems by Angelos Zoumis.

## Code
In order to run the code, python 3 and numpy is required. Furthermore, the graphs can be downloaded from here: https://pacechallenge.org/2021/tracks

The distance metrics are reversed, meaning that small value indicates that two vertices are far, while a large value idicates that they are close to each other.

To change the source of the files, line 6 of Main.py can be changed.

To change the time given to the algorithms, line 8 of Main.py can be changed.

From line 10 to 32 in Main.py, the combination of algorithms and distance metrics that will be used are declared. The code will produce the file output.json, which will contain be in the following format:

{"\<algorithm\> \<distance metric\>": {"\<graph file name\>": {"\<time\>": \<edits\>, ...}, ...}, ...}

## Experiment output
The json file test-1.json contains the results of the test with the 6 distance metrics combined with the 3 different algorithms. test-2.json contains the results from the experiments using different i for the k-means algorithm.
