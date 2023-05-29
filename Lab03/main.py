import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

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

def generate_graph(min_vert, max_vert):
    nodes = random.randint(min_vert, max_vert)
    degrees = np.array([random.randint(2, nodes-1) for _ in range(nodes)])
    while not is_seq_graphical(degrees):
        degrees = np.array([random.randint(1, nodes-1) for _ in range(nodes)])
    adjacency_matrix = build_graph_from_degrees(degrees)

    return adjacency_matrix

def is_graph_compact(component_list):
    num_of_comp = np.unique(component_list).size
    if num_of_comp == 1:
        return True
    return False

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

def __connected_component_recursive(nr, index, adjacency_matrix, comp):
    neighbours = []
    for index, node in enumerate(adjacency_matrix[index]):
        if node == 1:
            neighbours.append(index)

    for node in neighbours:
        if comp[node] == -1:
            comp[node] = nr
            __connected_component_recursive(nr, node, adjacency_matrix, comp)


def draw_graph_with_costs(adjacency_matrix, costs, title=''):
    _, ax = plt.subplots()
    circle = plt.Circle((0, 0), 1, fill=False, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax.add_patch(circle)
    ax.axis('equal')
    ax.set_title(title)
    G = nx.from_numpy_array(adjacency_matrix)
    nx.draw_circular(G, with_labels=True, ax=ax)
    
    costs_dict = dict()

    num_of_verticles = len(adjacency_matrix)
    k = 0
    for i in range(num_of_verticles-1):
        j = i + 1
        while j < num_of_verticles:
            if adjacency_matrix[i][j] == 1:
                costs_dict.update({(i, j) : costs[k]})
                k += 1
            j += 1

    num_of_edges = sum(sum(row) for row in adjacency_matrix) // 2
    c = random.choices(['r', 'g', 'b', 'y', 'm'], k=num_of_edges)
    nx.draw_networkx_edges(G, pos=nx.circular_layout(G), edge_color=c)
    nx.draw_networkx_edge_labels(G, pos=nx.circular_layout(G), edge_labels=costs_dict)
    plt.show()


def init(graph, root):
    num_of_verticles = len(graph)
    vert_data = dict({i : (float('inf'), None)  for i in range(num_of_verticles)})
    vert_data[root] = (0, None)

    return vert_data


def relax(conn, weight, u, v):
    if conn[v][0] > conn[u][0] + weight:
        conn[v] = (conn[u][0] + weight, u)


def dijkstra(adjacency_matrix, weights_list, root):
    vert_data = init(adjacency_matrix, root)
    vert_done = []

    while len(vert_done) < len(vert_data):
        u = min(list(filter(lambda x: x[0] not in vert_done, vert_data.items())), key=lambda x: x[1][0])[0]
        vert_done.append(u)        
        neighbourhood = [i for i, v in enumerate(adjacency_matrix[u]) if v == 1]
        
        for neigh in neighbourhood:
            if neigh not in vert_done:
                relax(vert_data, weights_list[frozenset([u, neigh])], u, neigh)
    return vert_data


def adjacency_matrix_with_weights(adjacency_matrix, edge_cost):
    edgeCounter = 0
    adjacency_matrix_with_weights = [[0 for i in adjacency_matrix] for j in adjacency_matrix]
    for rowIndex, row in enumerate(adjacency_matrix):
        for columnIndex, column in enumerate(row):
            if column == 1 and edgeCounter < len(edge_cost) and columnIndex > rowIndex:
                adjacency_matrix_with_weights[rowIndex][columnIndex] = edge_cost[edgeCounter]
                adjacency_matrix_with_weights[columnIndex][rowIndex] = edge_cost[edgeCounter]
                edgeCounter += 1

    return adjacency_matrix_with_weights

def prima(adjacency_matrix_with_weights):
    INF = 9999999
    # number of vertices in graph
    N = len(adjacency_matrix_with_weights)
    selected_node = [0 for i in range(N)]
    no_edge = 0

    selected_node[0] = True

    # printing for edge and weight
    print("Edge : Weight\n")
    while (no_edge < N - 1):

        minimum = INF
        a = 0
        b = 0
        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if ((not selected_node[n]) and adjacency_matrix_with_weights[m][n]):
                        # not in selected and there is an edge
                        if minimum > adjacency_matrix_with_weights[m][n]:
                            minimum = adjacency_matrix_with_weights[m][n]
                            a = m
                            b = n
        print(str(a) + "-" + str(b) + ":" + str(adjacency_matrix_with_weights[a][b]))
        selected_node[b] = True
        no_edge += 1


if __name__ == '__main__':
    type = int(input('1.Random graph with costs\n2.Dijkstra algorithm \n3. Minimal spanning tree\nOption: '))
    print('')

    if type == 1:
        adjacency_matrix = generate_graph(4, 10)
        while not is_graph_compact(connected_component(adjacency_matrix)):
            adjacency_matrix = randomise_graph(adjacency_matrix, 1)
        num_of_edges = sum(sum(row) for row in adjacency_matrix) // 2

        edge_cost = [random.randint(1, 10) for _ in range(num_of_edges)]
        print('Edge costs: ', edge_cost)

        draw_graph_with_costs(adjacency_matrix, edge_cost, title='Random graph with weigths')

    elif type == 2:
        adjacency_matrix = generate_graph(8, 8)
        while not is_graph_compact(connected_component(adjacency_matrix)):
            adjacency_matrix = randomise_graph(adjacency_matrix, 1)
        num_of_edges = sum(sum(row) for row in adjacency_matrix) // 2

        edge_cost = [random.randint(1, 10) for _ in range(num_of_edges)]
        costs_dict = dict()
        num_of_verticles = len(adjacency_matrix)
        max_index = num_of_verticles-1
        k = 0

        for i in range(num_of_verticles-1):
            j = i + 1
            while j < num_of_verticles:
                if adjacency_matrix[i][j] == 1:
                    costs_dict.update({frozenset([i, j]) : edge_cost[k]})
                    k += 1
                j += 1
        
        for i in range(max_index+1):
            d = dijkstra(adjacency_matrix, costs_dict, i)
            for j in d.items():
                path = [ j[0] ]
                next_hop = j[1][1]
                while next_hop != None:
                    path.append(next_hop)
                    next_hop = d[next_hop][1]
                path = path[::-1]
                print(f'{i}->{j[0]}: ', path)

        # Ex. 3
        matrix = []
        print(costs_dict)
        for i in range(max_index+1):
            d = dijkstra(adjacency_matrix, costs_dict, i)
            tmp = []
            print(d.items())
            for j in d.items():
                path_sum = j[1][0]
                tmp.append(path_sum)
            matrix.append(tmp)
        matrix = np.array(matrix)
        print(matrix)

        # Ex. 4
        graph_center = min(enumerate(np.sum(matrix, axis=1)), key=lambda x: x[1])[0]
        min_max_center = min(enumerate(np.max(matrix, axis=1)), key=lambda x: x[1])[0]
        print('Center: ', graph_center)
        print('MinMaxCenter: ', min_max_center)

        draw_graph_with_costs(adjacency_matrix, edge_cost, title='Random graph with weigths')

    # Ex. 5
    elif type == 3:
        adjacency_matrix = generate_graph(8, 8)
        while not is_graph_compact(connected_component(adjacency_matrix)):
            adjacency_matrix = randomise_graph(adjacency_matrix, 1)
        num_of_edges = sum(sum(row) for row in adjacency_matrix) // 2
        edge_cost = [random.randint(1, 10) for _ in range(num_of_edges)]
        adjacencyMatrixWithWeights = adjacency_matrix_with_weights(adjacency_matrix, edge_cost)
        prima(adjacencyMatrixWithWeights)
        draw_graph_with_costs(adjacency_matrix, edge_cost, title='Random graph with weigths')

    else:
        print('Wrong option.')
