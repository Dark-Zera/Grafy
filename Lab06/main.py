import numpy as np
import random as rd
import copy
from string import ascii_lowercase
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


def simulated_annealing(adjacency_list):
	# wyznacz dowolny cykl startowy P
	P = []
	P_new = []
	d = lambda x: x # uzupelnic

	max_iter = 10000
	for i in range(100, 0, -1):
		T = 0.001 * i**2
		for it in range(max_iter):
			# wylosuj ab cd z P
			# nowy cykl P
			# zamiana ab i cd
			if d(P_new) < d(P):
				P = P_new.copy()
			else:
				r = rd.random()
				if r < np.exp(-(d(P_new) - d(P))/T):
					P = P_new.copy()
	return P

if __name__ == '__main__':
	op = int(input("1. Provide graph from file\n2. Generate random graph\n3. Annealing\n"))
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
	
	# Niepotrzebne
	elif op == 2:
		n = 10
		k = 4
		adjacency_matrix = np.array(generate_random_regular_graph(n, k))
		adjacency_list = np.array(adjacency_matrix_to_adjacency_list(adjacency_matrix)) - 1
	
	elif op == 3:
		pass
	
	pr_result = random_page_rank(adjacency_list)
	print_pg_sorted(pr_result)
	print('=====================================')
	pr_result, _ = page_rank(adjacency_list)
	print_pg_sorted(pr_result)