import numpy
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

def draw_graph(adjacency_matrix, title=''):
    _, ax = plt.subplots()
    circle = plt.Circle((0, 0), 1, fill=False, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax.add_patch(circle)
    ax.axis('equal')
    ax.set_title(title)
    G = nx.from_numpy_array(adjacency_matrix)
    nx.draw_circular(G, with_labels=True, ax=ax)
    plt.show()

def adjacency_matrix_to_adjacency_list(matrix):
    adjacency_list = []
    for row in matrix:
        adjacency_list.append([])
        for i in range(len(row)):
            if row[i] == 1:
                adjacency_list[-1].append(i+1)

    return adjacency_list


def is_seq_graphical(seq):
    seq = np.sort(seq.flatten())
    seq = seq[::-1]

    if np.take(seq, 0) > seq.size or np.sum(seq) % 2:
        return False

    while True:
        max_value = np.take(seq, 0)

        if np.take(seq, -1) < 0:
            return False
        elif max_value == 0:
            return True

        for i in range(1, max_value + 1):
            seq[i] -= 1

        seq.itemset(0, 0)
        seq = np.sort(seq)
        seq = seq[::-1]


def build_graph_from_degrees(seq):
    seq = np.sort(seq.flatten())
    seq = seq[::-1]
    degrees = dict(enumerate(seq))
    num_of_verticles = seq.size
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


def randomise_graph(graph, repeat):
    rand_graph = np.copy(graph)
    while repeat:
        edges = np.where(rand_graph == 1)
        t_edges = list(zip(edges[0], edges[1]))

        e1, e2 = random.sample(t_edges, 2)
        new_e1, new_e2 = (e1[0], e2[1]), (e2[0], e1[1])

        if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
            continue
        if (new_e1 not in t_edges) and (new_e2 not in t_edges):
            rand_graph.itemset((e1[0], e1[1]), 0)
            rand_graph.itemset((e2[0], e2[1]), 0)
            rand_graph.itemset((e1[1], e1[0]), 0)
            rand_graph.itemset((e2[1], e2[0]), 0)
            rand_graph.itemset((new_e1[0], new_e1[1]), 1)
            rand_graph.itemset((new_e2[0], new_e2[1]), 1)
            rand_graph.itemset((new_e1[1], new_e1[0]), 1)
            rand_graph.itemset((new_e2[1], new_e2[0]), 1)
            repeat -= 1
    return rand_graph


def is_graph_compact(component_list):
    num_of_comp = np.unique(component_list).size
    if num_of_comp == 1:
        return True
    return False


def generate_euler_graph():
    adjacency_matrix = generate_graph(4, 10)
    while any([sum(vert) % 2 == 1 for vert in adjacency_matrix]):
        adjacency_matrix = generate_graph(4, 10)
    while not is_graph_compact(connected_component(adjacency_matrix)):
        adjacency_matrix = randomise_graph(adjacency_matrix, 1)

    return adjacency_matrix


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

def generate_graph(min_vert, max_vert):
    nodes = random.randint(min_vert, max_vert)
    degrees = np.array([random.randint(1, nodes-1) for _ in range(nodes)])
    while not is_seq_graphical(degrees):
        degrees = np.array([random.randint(1, nodes-1) for _ in range(nodes)])
    adjacency_matrix = build_graph_from_degrees(degrees)

    return adjacency_matrix


def is_graph_hamiltonian_quick_check(adjacency_matrix):
    if not is_graph_compact(connected_component(adjacency_matrix)):
        return False
    num_of_verticles = len(adjacency_matrix)
    if num_of_verticles >= 3 and all([sum(vert) >= num_of_verticles/2 for vert in adjacency_matrix]):
        return True
    
    return None


def find_hamiltonian_cycle(adjacency_matrix):
    adjacency_list = adjacency_matrix_to_adjacency_list(adjacency_matrix)
    for i in range(len(adjacency_list)):
        for j in range(len(adjacency_list[i])):
            adjacency_list[i][j] -= 1

    path = [0]
    visited = set([0])
    start_vertex = 0
    cycle = backtrack(start_vertex, adjacency_list, visited, path)

    if cycle is None:
        return None

    cycle.append(0)
    return cycle


def backtrack(vertex, adjacency_list, visited, path):
    if len(visited) == len(adjacency_list) and vertex in adjacency_list[0]:
        return path

    if vertex >= len(adjacency_list):
        vertex = len(adjacency_list) - 1

    for neighbor in adjacency_list[vertex]:
        if neighbor not in visited:
            visited.add(neighbor)
            path.append(neighbor)
            cycle = backtrack(neighbor, adjacency_list, visited, path)

            if cycle is not None:
                return cycle

            visited.remove(neighbor)
            path.pop()

    return None


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
    num_of_verticles = len(adjacency_matrix)
    comp = [-1] * num_of_verticles

    for index in range(num_of_verticles):
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


def generate_random_regular_graph(n, k):
    if k > n:
        return None
    elif k % 2 == 1 and n % 2 == 1:
        return None
    else:
        degrees = [k] * n
        matrix = build_graph_from_degrees(np.array(degrees))
        while not is_graph_compact(connected_component(matrix)):
            matrix = randomise_graph(matrix, 1)

        return matrix


if __name__ == '__main__':
    option = int(input('1. Graph based on degrees\n2. Euler\n3. k-regular graph\n4. Hamilton\nOption: '))
    print('')

    if option == 1:
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
        rand_adjacency_matrix = randomise_graph(adjacency_matrix, 100)
        print('Largest connected component list: ', largest_connected_component(adjacency_matrix))

        _, axies = plt.subplots(1, 2)

        draw_graph_with_subplots(adjacency_matrix, axies[0], 'Graph from data')
        draw_graph_with_subplots(rand_adjacency_matrix, axies[1], 'Randomise graph')
        plt.show()

    elif option == 2:
        adjacency_matrix = generate_euler_graph()
        print("Euler cycle for generated graph: ", fleury_algorithm(adjacency_matrix))
        draw_graph(adjacency_matrix, 'Randomised euler graph')

    elif option == 3:
        fun = input("Enter two values, first - number of nodes, second - degree\n").split(' ')
        adjacency_matrix = generate_random_regular_graph(int(fun[0]), int(fun[1]))
        # print(type(adjacency_matrix))
        if type(adjacency_matrix) is numpy.ndarray:
            draw_graph(adjacency_matrix, 'Hamiltonian graph')
        else:
            print("Cannot create k-regular graph with those parameters")

    elif option == 4:
        adjacency_matrix = generate_graph(8, 8)
        quick_check = is_graph_hamiltonian_quick_check(adjacency_matrix)
        if quick_check == False:
            print('Graph is hamiltonian: ', quick_check)
        else:
            cycle = find_hamiltonian_cycle(adjacency_matrix)
            if cycle is not None:
                print('Graph is hamiltonian: ', True)
                print("Hamilton cycle: ", cycle)
            else:
                print('Graph is hamiltonian: ', False)

        draw_graph(adjacency_matrix, 'Hamiltonian graph')

    else:
        print('Wrong option.')

