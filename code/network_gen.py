import networkx as nx
import argparse
import numpy as np
import os

def gen_model(N, p, model):
    """
    Generate a network using a single-parameter graph model.
    
    Parameters
    ----------
        N: int
            network size
        p: str
            input parameter for a single-parameter graph
        model: str
            which graph model to use
    """
    # choose graph model
    if model == 'ER':
        G = nx.erdos_renyi_graph(N, float(p))
    elif model == 'BA':
        p = int(p)
        g = nx.complete_graph(p)
        G = nx.barabasi_albert_graph(N, p, initial_graph=g)

    # save to file
    nx.write_pajek(G, f'./graphs/pajek/{model}/{N} {''.join(p.split('.'))}.net.gz')

def gen_graph_samples(N, p, n, model):
    """
    Generate samples of a network model given an input parameter.
    
    Parameters
    ----------
        N: int
            network size
        p: str
            input parameter point
        n: str
            number of networks to produce for each parameters
        model: str
            network model to generate
    """
    # generate the graph for each point in parameter space n times
    for i in range(n):
        # generate the network
        if model == 'ER':
            G = nx.erdos_renyi_graph(N, float(p))

        # save to file
        output = f'./graphs/pajek/{model}/{i}'
        os.makedirs(output, exist_ok=True)
        nx.write_pajek(G, f'{output}/{N} {''.join(p.split('.'))}.net.gz')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=5000, help='network size')
    parser.add_argument('-n', type=int, default=1, help='number of graphs to produce per parameter point')
    parser.add_argument('-p', type=str, help='probability of connecting two random nodes')
    parser.add_argument('-f', type=str, help='file containing graph parameter input')
    parser.add_argument('--gen', action='store_true', help='generate a graph')
    parser.add_argument('--samples', action='store_true', help='generate graph samples for given parameter point')
    parser.add_argument('-m', type=str, choices=['ER', 'BA'], help='Choose network model to generate')
    args = parser.parse_args()

    if args.gen:
        gen_model(args.N, args.p, args.m)
    elif args.samples:
        gen_graph_samples(args.N, args.p, args.n, args.m)