import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

from Lab01.main import draw_graph


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
    nodes = random.randint(4, 10)
    degrees = np.array([random.randint(1, int((nodes - 1) / 2)) * 2 for i in range(nodes)])

    adjacency_matrix = build_graph_from_degrees(degrees)
    randomise_graph(adjacency_matrix, random.randint(0, 10))

    print("Euler cycle for generated graph: ", fleury_algorithm(adjacency_matrix))
    draw_graph(adjacency_matrix, 'Randomised euler graph')


def fleury_algorithm(adjacency_matrix):
    temp_adjacency_matrix = np.copy(adjacency_matrix)
    n = len(temp_adjacency_matrix)
    current_node = 0
    circuit = [current_node]
    edges_left = sum(sum(row) for row in temp_adjacency_matrix) // 2

    while edges_left > 0:
        next_node = None

        for i in range(n):
            if temp_adjacency_matrix[current_node][i] == 1:
                disconnected = True
                temp_adjacency_matrix[current_node][i] = 0
                temp_adjacency_matrix[i][current_node] = 0

                for j in range(n):
                    if temp_adjacency_matrix[i][j] == 1:
                        disconnected = False
                        break

                if not disconnected:
                    next_node = i
                    break
                else:
                    temp_adjacency_matrix[current_node][i] = 1
                    temp_adjacency_matrix[i][current_node] = 1

        if next_node is None:
            if edges_left == 1:
                circuit.append(0)
            else:
                circuit.pop()
                current_node = circuit[-1]
        else:
            circuit.append(next_node)
            current_node = next_node

        edges_left -= 1

    return circuit


def draw_graph_with_subplots(matrix, ax, title=''):
    circle = plt.Circle((0, 0), 1, fill=False, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax.add_patch(circle)
    G = nx.from_numpy_array(matrix)
    ax.axis('equal')
    ax.set_title(title)
    nx.draw_circular(G, with_labels=True, ax=ax)


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


def print_graph_from_adjacency_matrix(adjacency_matrix):
    G = nx.from_numpy_array(np.array(adjacency_matrix))
    nx.draw_circular(G, with_labels=True)
    plt.axis('equal')
    plt.show()


def generate_random_regular_graph(n, k):
    if k > n:
        return None
    elif k % 2 == 1 and n % 2 == 1:
        return None
    else:
        degrees = [k for node in range(n)]
        matrix = build_graph_from_degrees(np.array(degrees))
        matrix = randomise_graph(matrix, 5)
        return matrix


if __name__ == '__main__':
    type = int(input('1. Graph based on degrees\n2. Euler\n3. Hamilton\nOption: '))
    print('')

    if type == 1:
        path = 'data/'
        path += input(
            'Enter path to file containing graph data in form of list of degrees.\nPath: ')
        print('')
        degrees_data = pd.read_csv(path, sep=' ', header=None).to_numpy()
        print('Data: ', degrees_data.flatten())

        if not is_seq_graphical(degrees_data):
            print('Graph isnt graphical.')
            quit()

        adjacency_matrix = build_graph_from_degrees(degrees_data)
        rand_adjacency_matrix = randomise_graph(adjacency_matrix, 10)
        print('Largest connected component list: ', largest_connected_component(adjacency_matrix))

        _, axies = plt.subplots(1, 2)

        draw_graph_with_subplots(adjacency_matrix, axies[0], 'Graph from data')
        draw_graph_with_subplots(rand_adjacency_matrix, axies[1], 'Randomise graph')

        plt.show()

    elif type == 2:
        generate_euler_graph()
    elif type == 3:
        pass
    else:
        print('Wrong option.')

