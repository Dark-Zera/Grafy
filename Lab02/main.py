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

		for i in range(1, max_value+1):
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
		for i in range(1, num_of_edges+1):
			degrees[keys[i]] -= 1
			adjacency_matrix.itemset((curr_ver, keys[i]),  1)
			adjacency_matrix.itemset((keys[i], curr_ver),  1)
		
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

if __name__ == '__main__':
	path = 'Lab02/data/'
	path += input('Enter path to file containing graph data in form of list of degrees.\n\
	       (ex. g1.txt ) {Format of data: 4 3 2 1}\n\t\tPath:')
	type = int(input('1.) Graph based on degrees\n2.) Euler\n3.) Hamilton\nOption:'))
	print('')
	
	if type == 1:
		degrees_data = pd.read_csv(path, sep=' ', header=None).to_numpy()
		print('Data: ', degrees_data.flatten())

		if not is_seq_graphical(degrees_data):
			print('Graph isnt graphical.')
			quit()

		adjacency_matrix = build_graph_from_degrees(degrees_data)
		rand_adjacency_matrix = randomise_graph(adjacency_matrix,  10)

		_, axies = plt.subplots(1, 2)

		# Long first loading
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

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