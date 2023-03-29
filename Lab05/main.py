import numpy as np
from Lab04.main import adjacency_list_to_adjacency_matrix
from Lab04.main import draw_digraph

if __name__ == '__main__':
	separator_line = int()
	data = list()
	graph_file_path = 'Lab05/data/' + 'g1.txt'

	with open(graph_file_path, "r") as fp:
		data = fp.readlines()
		for i, line in enumerate(data):
			if line == '\n':
				separator_line = i
				break

	adjacency_list = list()
	graph_weights = list()
	for line in data[:separator_line]:
		adjacency_list.append(list(map(int, line.strip().split())))
	for line in data[separator_line+1:-1]:
		graph_weights.append(list(map(int, line.strip().split())))


	adjacency_list.append([])
	graph_weights.append([])
	print('Delimiter: ', separator_line)
	print('List: ', adjacency_list)
	print('Weights: ', graph_weights)

	adjacency_matrix = adjacency_list_to_adjacency_matrix(adjacency_list)
	tmp = []
	for row in graph_weights:
		for column in row:
			tmp.append(column)
	graph_weights = tmp
	print(graph_weights)
	draw_digraph(adjacency_matrix, 'Digraph with weights', graph_weights)

	