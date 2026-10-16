import networkx as nx
import argparse
import numpy as np

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

def gen_ER_from_list(N, filename):
    """
    Generate a series of Erdos-Renyi networks from a list of 
    edge probability values.
    
    Arguments:
        N: network size
        filename: file with p values
    """
    # load in probabilities
    p_vals = np.loadtxt(filename, dtype=str)

    # iterate over each p value
    for p in p_vals:
        print(p)

        # generate the network
        G = nx.erdos_renyi_graph(N, float(p))

        # save to file
        nx.write_pajek(G, f'./graphs/pajek/ER/{N} {''.join(p.split('.'))}.net.gz')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=5000, help='network size')
    parser.add_argument('-p', type=float, default=0.1, help='probability of connecting two random nodes')
    parser.add_argument('-f', type=str, help='file containing graph parameter input')
    parser.add_argument('--ER', action='store_true', help='generate an ER random graph')
    parser.add_argument('--ER_list', action='store_true', help='generate ER random graphs from parameter list')
    args = parser.parse_args()

    if args.ER:
        gen_ER_model(args.N, args.p)
    elif args.ER_list:
        gen_ER_from_list(args.N, args.f)