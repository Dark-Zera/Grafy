import numpy as np
import random as rd
import copy
import matplotlib.pyplot as plt
from string import ascii_lowercase
from numba import jit

from Lab01.main import generate_random_graph_nodes_lines, draw_graph
from Lab04.main import adjacency_matrix_to_adjacency_list, adjacency_list_to_adjacency_matrix
from Lab02.main import generate_random_regular_graph

# adjacency_list - [0, max-1]

def page_rank(adjacency_list):
	adjacency_matrix = adjacency_list_to_adjacency_matrix([[i+1 for i in l] for l in adjacency_list])
	teleport_probability = 0.15
	matrix_size = adjacency_matrix.shape[0]
	probability = np.array([1./matrix_size] * (matrix_size))
	err = 1.
	it = 0

	page_rank_matrix = copy.deepcopy(adjacency_matrix)
	out_rank = np.sum(adjacency_matrix, axis=1)


	tmp = np.zeros(shape=(matrix_size, matrix_size))
	for i, l  in enumerate(page_rank_matrix):
		for j, v in enumerate(l):
			tmp[i][j] = ((v * (1.-teleport_probability)) / (out_rank[i])) + (teleport_probability / matrix_size)
	page_rank_matrix = np.array(tmp)
	
	while err > 1e-6:
		old_probability = probability.copy()
		probability = probability.dot(page_rank_matrix)
		it += 1
		err = np.linalg.norm(probability - old_probability)
	
	return probability, it

def random_page_rank(adjacency_list):
	teleport_probability = 0.15
	N = 100000
	probability = [0.] * (adjacency_list.shape[0])

	curr_vert = rd.randrange(adjacency_list.shape[0])
	probability[curr_vert] += 1

	for _ in range(N):
		action = rd.choices([0, 1], weights=[teleport_probability, 1-teleport_probability])
		if action == 1:
			new_vert = rd.randrange(adjacency_list.shape[0])
			while new_vert == curr_vert:
				new_vert = rd.randrange(adjacency_list.shape[0])
			curr_vert = new_vert
		else:
			new_vert = rd.choice(adjacency_list[curr_vert])
			while new_vert == curr_vert:
				new_vert = rd.choice(adjacency_list[curr_vert])
			curr_vert = new_vert
		
		probability[curr_vert] += 1

	probability = list(map(lambda x : x/N, probability))
	return probability


def print_pg_sorted(pg_in_order):
	l = list(ascii_lowercase[:len(pg_in_order)])
	pg_in_order = {l[i] : pg_in_order[i] for i in range(len(pg_in_order))}
	pg_in_order = dict(sorted(pg_in_order.items() , key=lambda x: x[1], reverse=True))

	for k, v in pg_in_order.items():
		print(f'{k} ==> PageRank = {v}')

@jit(nopython=True)
def simulated_annealing(P):
	max_iter = 30000
	for i in range(100, 0, -1):
		T = 0.001 * i**2
		for it in range(max_iter):
			[a_index, b_index, c_index, d_index] = np.random.choice(len(P), 4, False)
			a = P[a_index].copy()
			b = P[b_index].copy()
			c = P[c_index].copy()
			d = P[d_index].copy()

			new_cycle = P.copy()

			b[0] = c_index
			c[0] = b_index
			new_cycle[b_index] = c.copy()
			new_cycle[c_index] = b.copy()

			P_new = new_cycle.copy()

			old_cost = 0
			for index, node in enumerate(P):
				if index + 1 < len(P):
					old_cost += abs(P[index + 1][1] - P[index][1])
					old_cost += abs(P[index + 1][2] - P[index][2])
				else:
					old_cost += abs(P[0][1] - P[index][1])
					old_cost += abs(P[0][2] - P[index][2])

			new_cost = 0
			for index, node in enumerate(P_new):
				if index + 1 < len(P_new):
					new_cost += abs(P_new[index + 1][1] - P_new[index][1])
					new_cost += abs(P_new[index + 1][2] - P_new[index][2])
				else:
					new_cost += abs(P_new[0][1] - P_new[index][1])
					new_cost += abs(P_new[0][2] - P_new[index][2])

			if new_cost < old_cost:
				P = P_new.copy()
			else:
				r = rd.random()
				if r < np.e**(-(new_cost - old_cost)/T):
					P = P_new.copy()
	return P

def draw_chess_board(cycle):
	x = [node[1] for node in cycle]
	y = [node[2] for node in cycle]
	last_connection_x = [x[0], x[-1]]
	last_connection_y = [y[0], y[-1]]

	plt.scatter(x, y)
	plt.plot(x, y, 'b-')
	plt.plot(last_connection_x, last_connection_y, 'b-')

	plt.xlabel("x")
	plt.ylabel("y")
	plt.show()

if __name__ == '__main__':
	op = int(input("1. Provide graph from file\n2. Annealing\n"))
	adjacency_list = []
	adjacency_matrix = []

	if op == 1:
		path = "Lab06/data/" + input("Please provide path to file\n")
		with open(path, 'r') as f:
			data = f.readlines()
			for line in data:
				adjacency_list.append(list(map(int, line.strip().split())))
			adjacency_list = np.array(adjacency_list, dtype=object)
			#adjacency_matrix = adjacency_list_to_adjacency_matrix([[i+1 for i in l] for l in adjacency_list])

	elif op == 2:
		node_list = []
		with open("data/input_150.txt") as file:
			lines = file.readlines()
			for index, line in enumerate(lines):
				position = [float(i) for i in line.split(" ")]
				node_list.append([index, position[0], position[1]])

		P = np.array([np.array([index, node_list[index][1], node_list[index][2]]) for index in range(len(node_list))])

		result_cycle = simulated_annealing(P)

		draw_chess_board(result_cycle)
