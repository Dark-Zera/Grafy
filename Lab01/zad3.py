import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from main import incidence_matrix_to_adjacency_matrix

def generateRandomGraphNodesLines(nodes, lines):
    incident_matrix = [[0 for i in range(lines)] for j in range(nodes)]

    for column in range(lines):
        selectedNodes = random.choices(range(nodes), k=2)
        incident_matrix[selectedNodes[0]][column] = 1
        incident_matrix[selectedNodes[1]][column] = 1

    G = nx.from_numpy_array(incidence_matrix_to_adjacency_matrix(incident_matrix))
    nx.draw_circular(G, with_labels=True)
    plt.axis('equal')
    plt.show()

    return incident_matrix


def generateRandomGraphNodesProbability(nodes, probability):
    adjacency_matrix = [[0 for i in range(nodes)] for j in range(nodes)]
    for row in range(nodes):
        for column in range(row):
            if random.choices([0, 1], weights=[1-probability, probability])[0] == 1:
                adjacency_matrix[row][column] = 1
                adjacency_matrix[column][row] = 1

    G = nx.from_numpy_array(np.array(adjacency_matrix))
    nx.draw_circular(G, with_labels=True)
    plt.axis('equal')
    plt.show()

    return adjacency_matrix


generateRandomGraphNodesLines(15, 10)