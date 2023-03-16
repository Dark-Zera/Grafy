import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

from Lab01.main import draw_graph, adjacency_matrix_to_adjacency_list
from Lab02.main import generate_graph, is_graph_compact, randomise_graph, build_graph_from_degrees, is_seq_graphical


if __name__ == '__main__':
    type = int(input('1. Graph based on degrees\n2. Euler\n3. Hamilton\nOption: '))
    print('')

    if type == 1:
        path = 'data/'
        path += input(
            'Enter path to file containing graph data in form of list of degrees.\nPath: ')
        print('')
        degrees_data = pd.read_csv(path, sep=' ', header=None).to_numpy()
        print('Data: ', degrees_data.flatten())


    elif type == 2:
        pass
    elif type == 3:
        pass
    else:
        print('Wrong option.')
