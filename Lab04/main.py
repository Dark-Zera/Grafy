import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


def init(graph, root):
    num_of_verticles = len(graph)
    vert_data = dict({i: (float('inf'), None) for i in range(num_of_verticles)})
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
    num_edges = sum(sum(row) for row in adjacency_matrix)
    incidence_matrix = [[0 for _ in range(num_edges)] for _ in range(num_nodes)]
    edge_index = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            if adjacency_matrix[i][j] == 1:
                incidence_matrix[i][edge_index] = -1
                incidence_matrix[j][edge_index] = 1
                edge_index += 1
    return incidence_matrix


def incidence_matrix_to_adjacency_list(incidence_matrix):
    num_of_verticles = len(incidence_matrix)
    num_of_edges = len(incidence_matrix[0])
    adjacency_list = list([] for _ in range(num_of_verticles))

    for i in range(num_of_edges):
        nodes_in_edge = incidence_matrix[:, i]
        out_node = np.where(nodes_in_edge == 1)
        in_node = np.where(nodes_in_edge == -1)
        if in_node[0].size == 1 and out_node[0].size == 1:
            adjacency_list[in_node[0][0]].append(out_node[0][0] + 1)

    return np.array(adjacency_list)


def incidence_matrix_to_adjacency_matrix(incidence_matrix):
    num_of_verticles = len(incidence_matrix)
    num_of_edges = len(incidence_matrix[0])
    adjacency_matrix = [[0 for _ in range(num_of_verticles)] for _ in range(num_of_verticles)]

    for i in range(num_of_edges):
        nodes_in_edge = incidence_matrix[:, i]
        out_node = np.where(nodes_in_edge == 1)
        in_node = np.where(nodes_in_edge == -1)
        if in_node[0].size == 1 and out_node[0].size == 1:
            adjacency_matrix[in_node[0][0]][out_node[0][0]] = 1

    return np.array(adjacency_matrix)


def draw_digraph(adjacency_matrix, title='', weights=None):
    _, ax = plt.subplots()
    ax.axis('equal')
    ax.set_title(title)

    G = nx.DiGraph(directed=True)

    edges = []
    for r_id, row in enumerate(adjacency_matrix):
        for c_id, column in enumerate(row):
            if column == 1:
                edges.append((r_id, c_id))
    G.add_nodes_from(list(range(len(adjacency_matrix))))
    G.add_edges_from(edges)
    num_of_edges = sum(sum(row) for row in adjacency_matrix)
    c = random.choices(['r', 'g', 'b', 'y', 'm'], k=num_of_edges)

    nx.draw(G, pos=nx.fruchterman_reingold_layout(G), with_labels=True, ax=ax)

    if weights is not None:
        weights_dict = dict()
        num_of_verticles = len(adjacency_matrix)
        k = 0
        for i in range(num_of_verticles - 1):
            j = i + 1
            while j < num_of_verticles:
                if adjacency_matrix[i][j] == 1:
                    weights_dict.update({(i, j): weights[k]})
                    k += 1
                j += 1

        nx.draw_networkx_edge_labels(G, pos=nx.fruchterman_reingold_layout(G), edge_labels=weights_dict)
    plt.show()


def generate_random_graph_nodes_probability(nodes, probability):
    adjacency_matrix = [[0 for _ in range(nodes)] for _ in range(nodes)]

    for row in range(nodes):
        for column in range(row):
            if random.choices([0, 1], weights=[1 - probability, probability])[0] == 1:
                adjacency_matrix[row][column] = 1

    return np.array(adjacency_matrix)


def __connected_component_recursive(nr, index, adjacency_matrix, comp, d, f, iteration):
    neighbours = []
    for ind, node in enumerate(adjacency_matrix[index]):
        if node == 1:
            neighbours.append(ind)
    for node in neighbours:
        if comp[node] == -1:
            comp[node] = nr
            d[node] = iteration[0]
            iteration[0] += 1
            __connected_component_recursive(nr, node, adjacency_matrix, comp, d, f, iteration)
    f[index] = iteration[0]
    iteration[0] += 1


def connected_component(adjacency_matrix, search_list=None):
    nr = 0
    num_of_verticles = len(adjacency_matrix)
    comp = [-1] * num_of_verticles
    d = [-1] * num_of_verticles
    f = [-1] * num_of_verticles
    iteration = [1]  # reference

    node_list = list(range(num_of_verticles))
    if search_list:
        node_list = np.copy(search_list)

    for index in node_list:
        if comp[index] == -1:
            nr += 1
            comp[index] = nr
            d[index] = iteration[0]
            iteration[0] += 1
            __connected_component_recursive(nr, index, adjacency_matrix, comp, d, f, iteration)

    return comp, d, f


def kosaraju_algorithm(adjacency_matrix):
    tmp_matrix = np.copy(adjacency_matrix)
    _, _, f = connected_component(tmp_matrix)
    tmp_matrix = np.transpose(tmp_matrix)
    search_list = zip(range(len(adjacency_matrix)), f)
    search_list = sorted(search_list, key=lambda x: x[1], reverse=True)
    search_list = list(map(lambda x: x[0], search_list))
    # print(f)
    # print(search_list)
    conn2 = connected_component(tmp_matrix, search_list=search_list)
    return conn2


def draw_graph_v2(graph):
    G = nx.DiGraph()

    for src, neighbors in enumerate(graph):
        for dest, weight in neighbors:
            G.add_edge(src, dest, weight=weight)

    pos = nx.fruchterman_reingold_layout(G)

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.axis('off')
    plt.show()


def generate_random_graph(num_vertices, prob):
    graph = [[] for _ in range(num_vertices)]

    for src in range(num_vertices):
        for dest in range(num_vertices):
            if src != dest and random.random() < prob:
                graph[src].append((dest, random.randint(-1, 10)))

    return graph


def bellman_ford(graph, start):
    num_vertices = len(graph)

    distance = [float('inf')] * num_vertices
    distance[start] = 0

    for _ in range(num_vertices - 1):
        for src in range(num_vertices):
            for dest, weight in graph[src]:
                if distance[src] != float('inf') and distance[src] + weight < distance[dest]:
                    distance[dest] = distance[src] + weight

    for src in range(num_vertices):
        for dest, weight in graph[src]:
            if distance[src] != float('inf') and distance[src] + weight < distance[dest]:
                return False

    return distance


def johnson(graph):
    num_vertices = len(graph)
    augmented_graph = graph + [[(i, 0) for i in range(num_vertices)]]
    potential = bellman_ford(augmented_graph, num_vertices)

    if potential is False:
        print("Graph contains negative-weight cycle")
        return None

    reweighted_graph = [[] for _ in range(num_vertices)]

    for src in range(num_vertices):
        for dest, weight in graph[src]:
            reweighted_weight = weight + potential[src] - potential[dest]
            reweighted_graph[src].append((dest, reweighted_weight))

    all_shortest_paths = []
    for src in range(num_vertices):
        shortest_paths = bellman_ford(reweighted_graph, src)
        if shortest_paths is False:
            print("Graph contains negative-weight cycle")
            return None
        actual_shortest_paths = [distance - potential[src] + potential[dest] for dest, distance in
                                 enumerate(shortest_paths)]
        all_shortest_paths.append(actual_shortest_paths)

    return all_shortest_paths


if __name__ == '__main__':
    op = int(input(
        "1. Provide graph from file\n"
        "2. Generate graph with n nodes and probability of p.\n"
        "3. Kosaraju algorithm\n"
        "4. Bellman-Ford algorithm\n"
        "5. Johnson algorithm\n"))
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

        print("Adjacency matrix:")
        for row in adjacency_matrix:
            print(row)
        print("Adjacency list:")
        for row in adjacency_list:
            print(row)
        print("Incidence matrix:")
        for row in incident_matrix:
            print(row)
        draw_digraph(adjacency_matrix)

    elif op == 2:
        fun = input("Insert number of nodes and probability of connection.\n").split(' ')
        n = int(fun[0]) if len(fun) > 1 else 15
        p = float(fun[1]) % 1.0 if len(fun) > 2 else 0.3
        adjacency_matrix = generate_random_graph_nodes_probability(n, p)
        adjacency_list = adjacency_matrix_to_adjacency_list(adjacency_matrix)
        incident_matrix = adjacency_matrix_to_incidence_matrix(adjacency_matrix)

        print("Adjacency matrix:")
        for row in adjacency_matrix:
            print(row)
        print("Adjacency list:")
        for row in adjacency_list:
            print(row)
        print("Incidence matrix:")
        for row in incident_matrix:
            print(row)

        draw_digraph(adjacency_matrix)

    elif op == 3:
        path = "data/adjacencyMatrix.txt"
        adjacency_matrix = np.loadtxt(path).astype(int)
        # adjacency_matrix = generate_random_graph(7, 0.4)
        print("Adjacency matrix:")
        for row in adjacency_matrix:
            print(row)

        print("Kosaraju [#, d, f]: ", kosaraju_algorithm(adjacency_matrix))
        draw_digraph(adjacency_matrix)
    elif op == 4:
        start = int(input("Provide starting node.\n"))
        graph = generate_random_graph(5, 0.6)
        for index, i in enumerate(graph):
            print("wierzchołek", index, ":", i)

        print(bellman_ford(graph, start))
        draw_graph_v2(graph)
    elif op == 5:
        graph = generate_random_graph(5, 0.5)
        for index, i in enumerate(graph):
            print("wierzchołek", index, ":", i)

        for row in johnson(graph):
            for value in row:
                print("{:<6.0f}".format(value), end=' ')
            print()

        draw_graph_v2(graph)
    else:
        print("Wrong option selected.")
