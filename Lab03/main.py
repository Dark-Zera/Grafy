import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

from Lab02.main import generate_graph, is_graph_compact, randomise_graph, connected_component


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
        adjacency_matrix = generate_graph(4, 5)
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
        draw_graph_with_costs(adjacency_matrix, edge_cost, title='Random graph with weigths')
        prima(adjacencyMatrixWithWeights)
    else:
        print('Wrong option.')
