import random
import queue
import networkx as nx
import matplotlib.pyplot as plt


def generate_random_flow_network(N):
    G = nx.DiGraph()

    G.add_node('source', layer=0)
    G.add_node('target', layer=N + 1)

    for layer in range(1, N + 1):
        num_vertices = random.randint(2, N)
        for i in range(num_vertices):
            vertex = f'L{layer}-{i+1}'
            G.add_node(vertex, layer=layer)

    for layer in range(N + 1):
        current_layer_nodes = [node for node, attr in G.nodes(data=True) if attr['layer'] == layer]
        next_layer_nodes = [node for node, attr in G.nodes(data=True) if attr['layer'] == layer + 1]

        for node in current_layer_nodes:
            G.add_edge(node, random.choice(next_layer_nodes), capacity=random.randint(1, 10))
        for node in next_layer_nodes:
            G.add_edge(random.choice(current_layer_nodes), node, capacity=random.randint(1, 10))

    num_edges = 2 * N
    existing_edges = list(G.edges())

    while num_edges > 0:
        node1 = random.choice(list(G.nodes()))
        node2 = random.choice(list(G.nodes()))

        if node1 != 'target' and node2 != 'source' and node1 != node2 and (node1, node2) not in existing_edges:
            G.add_edge(node1, node2, capacity=random.randint(1, 10))
            num_edges -= 1

    return G


def draw_flow_network(G, show_flow = False):
    pos = nx.multipartite_layout(G, subset_key='layer')
    edge_labels = {(u, v): G[u][v]['capacity'] if not show_flow else str(G[u][v]['flow']) + '/' + str(G[u][v]['capacity']) for u, v in G.edges()}

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.axis('off')
    plt.show()


def bfs(graph, source, target, parent):
    visited = {node: False for node in graph.nodes}
    q = queue.Queue()
    q.put(source)
    visited[source] = True

    while not q.empty():
        node = q.get()

        for neighbor in graph.neighbors(node):
            if not visited[neighbor] and graph[node][neighbor]['capacity'] - graph[node][neighbor]['flow'] > 0:
                q.put(neighbor)
                visited[neighbor] = True
                parent[neighbor] = node

                if neighbor == target:
                    return True

    return False


def ford_fulkerson(graph, source, target):
    for u, v in graph.edges:
        graph[u][v]['flow'] = 0

    max_flow = 0
    iter = 0

    while iter < N * 100:
        parent = {node: None for node in graph.nodes}
        if not bfs(graph, source, target, parent):
            break

        path_flow = float('inf')
        v = target
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, graph[u][v]['capacity'] - graph[u][v]['flow'])
            v = u

        v = target
        while v != source:
            u = parent[v]
            if graph[u][v]:
                graph[u][v]['flow'] += path_flow
            else:
                graph[v][u]['flow'] -= path_flow
            v = u

        max_flow += path_flow
        iter += 1

    return max_flow


N = 4
flow_network = generate_random_flow_network(N)
max_flow = ford_fulkerson(flow_network, 'source', 'target')
print("Maksymalny przepływ:", max_flow)

draw_flow_network(flow_network, True)