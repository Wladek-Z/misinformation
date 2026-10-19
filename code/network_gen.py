import networkx as nx
import argparse
import numpy as np
import os

def gen_ER_model(N, p):
    """
    Generate a network using the Erdos-Renyi random graph model.
    
    Arguments:
        N: network size
        p: probability of connecting two randomly chosen nodes
    """
    # generate the network
    G = nx.erdos_renyi_graph(N, p)

    # save to file
    nx.write_pajek(G, f'./graphs/pajek/ER/{N} {''.join(str(p).split('.'))}.net.gz')

def gen_graph_from_1d_list(N, filename, n, model):
    """
    Generate a series of network models from a 1-dimensional list of 
    graph parameter values.
    
    Arguments:
        N: network size
        filename: file with parameter values
        n: number of networks to produce for each parameters
        model: network model to generate
    """
    # load in parameters
    params = np.loadtxt(filename, dtype=str)

    # iterate over each p value
    for p in params:
        print(p)

        # generate the graph for each point in parameter space n times
        for i in range(1, n+1):
            # generate the network
            if model == 'ER':
                G = nx.erdos_renyi_graph(N, float(p))

            # save to file
            output = f'./graphs/pajek/{model}/{i}'

            if not os.path.exists(output):
                os.makedirs(output)
                
            nx.write_pajek(G, f'{output}/{N} {''.join(p.split('.'))}.net.gz')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=5000, help='network size')
    parser.add_argument('-n', type=int, default=1, help='number of graphs to produce per parameter point')
    parser.add_argument('-p', type=float, default=0.1, help='probability of connecting two random nodes')
    parser.add_argument('-f', type=str, help='file containing graph parameter input')
    parser.add_argument('--ER', action='store_true', help='generate an ER random graph')
    parser.add_argument('--list', action='store_true', help='generate graphs from 1D parameter list')
    parser.add_argument('-m', type=str, choices=['ER'], help='Choose network model to analyse')
    args = parser.parse_args()

    if args.ER:
        gen_ER_model(args.N, args.p)
    elif args.list:
        gen_graph_from_1d_list(args.N, args.f, args.n, args.m)