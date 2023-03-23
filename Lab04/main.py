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
				adjacency_list[-1].append(i+1)

	return adjacency_list


def adjacency_list_to_adjacency_matrix(adjacency_list):
	adjacency_matrix = np.zeros((len(adjacency_list), len(adjacency_list)), dtype=int)
	index = 0
	for row in adjacency_list:
		for column in row:
			adjacency_matrix[index][column-1] = 1
		index += 1

	return adjacency_matrix

# do poprawy
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


def draw_digraph(adjacency_matrix, title=''):
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
	print(G)
	num_of_edges = sum(sum(row) for row in adjacency_matrix)
	c = random.choices(['r', 'g', 'b', 'y', 'm'], k=num_of_edges)
	
	#nx.draw_spring(G, with_labels=True)
	#nx.draw_networkx_edges(G, pos=nx.spring_layout(G), edge_color=c, arrows=True, arrowstyle='->')
	#nx.draw_networkx_nodes(G, pos=nx.spring_layout(G))
	nx.draw(G, pos=nx.fruchterman_reingold_layout(G), with_labels=True, ax=ax)
	plt.show()


def generate_random_graph_nodes_probability(nodes, probability):
	adjacency_matrix = [[0 for _ in range(nodes)] for _ in range(nodes)]
	
	for row in range(nodes):
		for column in range(row):
			if random.choices([0, 1], weights=[1-probability, probability])[0] == 1:
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
	iteration = [1] # reference

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
	#print(f)
	#print(search_list)
	conn2 = connected_component(tmp_matrix, search_list=search_list)
	return conn2

if __name__ == '__main__':

	op = int(input("1. Provide graph from file\n2. Generate graph with n nodes and probability of p.\n3. Kosaraju algorithm\n"))
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
		print("Adjacency matrix:")
		for row in adjacency_matrix:
			print(row)
		
		print("Kosaraju [#, d, f]: ", kosaraju_algorithm(adjacency_matrix))
	else:
		print("Wrong option selected.")

