import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


def is_seq_graphical(data):
    data = np.sort(data.flatten())
    data = data[::-1]

    if np.take(data, 0) > data.size or np.sum(data) % 2:
        return False

    while True:
        max_value = np.take(data, 0)

        if np.take(data, -1) < 0:
            return False
        elif max_value == 0:
            return True

        for i in range(1, max_value + 1):
            data[i] -= 1

        data.itemset(0, 0)
        data = np.sort(data)
        data = data[::-1]


def build_graph_from_degrees(data):
    data = np.sort(data.flatten())
    data = data[::-1]
    degrees = dict(enumerate(data))
    num_of_verticles = data.size
    adjacency_matrix = np.zeros((num_of_verticles, num_of_verticles), dtype=int)

    for _ in range(num_of_verticles):
        keys = list(degrees.keys())
        curr_ver = keys[0]
        num_of_edges = degrees[curr_ver]
        degrees[curr_ver] = 0
        for i in range(1, num_of_edges + 1):
            degrees[keys[i]] -= 1
            adjacency_matrix.itemset((curr_ver, keys[i]), 1)
            adjacency_matrix.itemset((keys[i], curr_ver), 1)

        degrees = dict(sorted(degrees.items(), key=lambda item: item[1], reverse=True))

    return adjacency_matrix


def randomise_graph(old_data, repeat):
    data = np.copy(old_data)
    while repeat:
        edges = np.where(data == 1)
        t_edges = list(zip(edges[0], edges[1]))

        e1, e2 = random.sample(t_edges, 2)
        new_e1, new_e2 = (e1[0], e2[1]), (e2[0], e1[1])

        if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
            continue
        if (new_e1 not in t_edges) and (new_e2 not in t_edges):
            data.itemset((e1[0], e1[1]), 0)
            data.itemset((e2[0], e2[1]), 0)
            data.itemset((e1[1], e1[0]), 0)
            data.itemset((e2[1], e2[0]), 0)
            data.itemset((new_e1[0], new_e1[1]), 1)
            data.itemset((new_e2[0], new_e2[1]), 1)
            data.itemset((new_e1[1], new_e1[0]), 1)
            data.itemset((new_e2[1], new_e2[0]), 1)
            repeat -= 1
    return data


def generate_euler_graph():
    nodes = random.randint(0, 10)
    degrees = [random.randint(1, 4) * 2 for i in range(nodes)]

    adjacency_matrix = build_graph_from_degrees(degrees)
    randomise_graph(adjacency_matrix, random.randint(0, 100))


def get_euler_cycle(adjacency_matrix):
    adjacency_matrix


def __connected_component_recursive(nr, index, adjacency_matrix, comp):
    neighbours = []
    for index, node in enumerate(adjacency_matrix[index]):
        if node == 1:
            neighbours.append(index)

    for node in neighbours:
        if comp[node] == -1:
            comp[node] = nr
            __connected_component_recursive(nr, node, adjacency_matrix, comp)


def connected_component(adjacency_matrix):
    nr = 0
    comp = [-1 for row in adjacency_matrix]

    for index, row in enumerate(adjacency_matrix):
        if comp[index] == -1:
            nr += 1
            comp[index] = nr
            __connected_component_recursive(nr, index, adjacency_matrix, comp)

    return comp


def largest_connected_component(adjacency_matrix):
    comp = connected_component(adjacency_matrix)

    largest_component = 0
    for node in comp:
        current_component = comp.count(node)
        if current_component > largest_component:
            largest_component = current_component

    largest_component_list = [index for index, node in enumerate(comp) if comp.count(node) == largest_component]

    return largest_component_list


if __name__ == '__main__':
    path = 'data/'
    path += input(
        'Enter path to file containing graph data in form of list of degrees.\n(ex. g1.txt ) {Format of data: 4 3 2 1}\n\t\tPath: ')
    print(path)
    type = int(input('1.) Graph based on degrees\n2.) Euler\n3.) Hamilton\nOption:'))
    print('')

    if type == 1:
        degrees_data = pd.read_csv(path, sep=' ', header=None).to_numpy()
        print('Data: ', degrees_data.flatten())

        if not is_seq_graphical(degrees_data):
            print('Graph isnt graphical.')
            quit()

        adjacency_matrix = build_graph_from_degrees(degrees_data)
        rand_adjacency_matrix = randomise_graph(adjacency_matrix, 10)

        _, axies = plt.subplots(1, 2)

        circle1 = plt.Circle((0, 0), 1, fill=False, color='r', linestyle='--', linewidth=2, alpha=0.5)
        axies[0].add_patch(circle1)
        G = nx.from_numpy_array(adjacency_matrix)
        axies[0].axis('equal')
        axies[0].set_title('Graph from data')
        nx.draw_circular(G, with_labels=True, ax=axies[0])

        circle2 = plt.Circle((0, 0), 1, fill=False, color='r', linestyle='--', linewidth=2, alpha=0.5)
        axies[1].add_patch(circle2)
        G2 = nx.from_numpy_array(rand_adjacency_matrix)
        axies[1].axis('equal')
        axies[1].set_title('Randomise graph')
        nx.draw_circular(G2, with_labels=True, ax=axies[1])

        plt.show()

    elif type == 2:
        pass
    elif type == 3:
        pass
    else:
        print('Wrong option.')
