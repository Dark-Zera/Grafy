import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


def adjacency_matrix_to_adjacency_list(matrix):
    adjacency_list = []
    for row in matrix:
        adjacency_list.append([])
        for i in range(len(row)):
            if row[i] == 1:
                adjacency_list[-1].append(i + 1)

    return adjacency_list


def adjacency_list_to_adjacency_matrix(adjacency_list):
    adjacency_matrix = np.zeros((len(adjacency_list), len(adjacency_list)), dtype=int)
    index = 0
    for row in adjacency_list:
        for column in row:
            adjacency_matrix[index][column - 1] = 1
        index += 1

    return adjacency_matrix


def adjacency_matrix_to_incidence_matrix(adjacency_matrix):
    num_nodes = len(adjacency_matrix)
    num_edges = sum(sum(row) for row in adjacency_matrix) // 2
    incidence_matrix = [[0 for j in range(num_edges)] for i in range(num_nodes)]
    edge_index = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i][j] == 1:
                incidence_matrix[i][edge_index] = 1
                incidence_matrix[j][edge_index] = 1
                edge_index += 1
    return incidence_matrix


def incidence_matrix_to_adjacency_list(incidence_matrix):
    num_of_verticles = len(incidence_matrix)
    num_of_edges = len(incidence_matrix[0])
    adjacency_list = list([] for i in range(num_of_verticles))

    for i in range(num_of_edges):
        edges = incidence_matrix[:, i]
        found = np.where(edges == 1)
        if found[0].size > 1:
            # Simple graph ( size = 2 )
            adjacency_list[found[0][0]].append(found[0][1] + 1)
            adjacency_list[found[0][1]].append(found[0][0] + 1)

    return np.array(adjacency_list)


def incidence_matrix_to_adjacency_matrix(incidence_matrix):
    num_of_verticles = len(incidence_matrix)
    num_of_edges = len(incidence_matrix[0])
    adjacency_matrix = [[0 for j in range(num_of_verticles)] for i in range(num_of_verticles)]

    for i in range(num_of_edges):
        edges = incidence_matrix[:, i]
        found = np.where(edges == 1)
        if found[0].size > 1:
            # Simple graph ( size = 2 )
            adjacency_matrix[found[0][0]][found[0][1]] = 1
            adjacency_matrix[found[0][1]][found[0][0]] = 1

    return np.array(adjacency_matrix)


def draw_graph(adjacency_matrix, title=''):
    _, ax = plt.subplots()
    circle = plt.Circle((0, 0), 1, fill=False, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax.add_patch(circle)
    ax.axis('equal')
    ax.set_title(title)
    G = nx.from_numpy_array(adjacency_matrix)
    nx.draw_circular(G, with_labels=True, ax=ax)
    plt.show()


def generate_random_graph_nodes_lines(nodes, lines):
    # Dodane po zajęciach:
    if lines > nodes * (nodes - 1) / 2:
        raise Exception("Number of lines is too big")

    incident_matrix = [[0 for i in range(lines)] for j in range(nodes)]

    for column in range(lines):
        nodes_already_selected = True
        while nodes_already_selected:
            selected_nodes = random.sample(range(nodes), k=2)
            for already_selected_nodes in range(column + 1):
                if incident_matrix[selected_nodes[0]][already_selected_nodes] == 1 and \
                        incident_matrix[selected_nodes[1]][already_selected_nodes] == 1:
                    nodes_already_selected = True
                    break
            else:
                nodes_already_selected = False

        incident_matrix[selected_nodes[0]][column] = 1
        incident_matrix[selected_nodes[1]][column] = 1

    return incidence_matrix_to_adjacency_matrix(np.array(incident_matrix))


def generate_random_graph_nodes_probability(nodes, probability):
    adjacency_matrix = [[0 for i in range(nodes)] for j in range(nodes)]
    for row in range(nodes):
        for column in range(row):
            if random.choices([0, 1], weights=[1 - probability, probability])[0] == 1:
                adjacency_matrix[row][column] = 1
                adjacency_matrix[column][row] = 1

    return np.array(adjacency_matrix)


if __name__ == '__main__':

    op = int(input("1. Provide graph from file\n2. Generate random graph\n"))
    if op == 1:
        path = "data/" + input("Please provide path to file\n")
        type = int(input(
            "Please provide type number according to the type of data in file\n1 - Adjacency matrix\n2 - Adjacency list\n3 - Incident matrix\n"))

        adjacency_matrix = []
        adjacency_list = []
        incident_matrix = []

        # if provided file contains adjacent matrix
        if type == 1:
            adjacency_matrix = np.loadtxt(path).astype(int)
            adjacency_list = adjacency_matrix_to_adjacency_list(adjacency_matrix)
            incident_matrix = adjacency_matrix_to_incidence_matrix(adjacency_matrix)

        # if provided file contains adjacency list
        elif type == 2:
            with open(path, 'r') as f:
                data = f.readlines()

            adjacency_list = []
            for line in data[:12]:
                adjacency_list.append(list(map(int, line.strip().split())))

            adjacency_matrix = adjacency_list_to_adjacency_matrix(adjacency_list)
            incident_matrix = adjacency_matrix_to_incidence_matrix(adjacency_matrix)

        # if provided file contains incident Matrix
        elif type == 3:
            incident_matrix = np.loadtxt(path).astype(int)
            adjacency_list = incidence_matrix_to_adjacency_list(incident_matrix)
            adjacency_matrix = incidence_matrix_to_adjacency_matrix(incident_matrix)

        else:
            print("Unknown number provided :(")
    elif op == 2:
        fun = input(
            "1. Generate graph with n nodes and k connections.\n2. Generate graph with n nodes and probability of p.\n(First value - option, Second and third value - function arguments (optional)\n ").split(
            ' ')
        if int(fun[0]) == 1:
            n = int(fun[1]) if len(fun) > 1 else 15
            k = int(fun[2]) if len(fun) > 2 else 10
            adjacency_matrix = generate_random_graph_nodes_lines(n, k)
        elif int(fun[0]) == 2:
            n = int(fun[1]) if len(fun) > 1 else 15
            p = float(fun[2]) % 1.0 if len(fun) > 2 else 0.3
            adjacency_matrix = generate_random_graph_nodes_probability(n, p)
        else:
            print("Wrong generator selected.")
        adjacency_list = adjacency_matrix_to_adjacency_list(adjacency_matrix)
        incident_matrix = adjacency_matrix_to_incidence_matrix(adjacency_matrix)
    else:
        print("Wrong option selected.")

    print("Adjacency matrix:")
    for row in adjacency_matrix:
        print(row)
    print("Adjacency list:")
    for row in adjacency_list:
        print(row)
    print("Incidence matrix:")
    for row in incident_matrix:
        print(row)

    draw_graph(adjacency_matrix)
