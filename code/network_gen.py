import networkx as nx
import argparse

def gen_ER_model(N, p, filename):
    """
    Generate a network using the Erdos-Renyi random graph model.
    
    Arguments:
        N: network size
        p: probability of connecting two randomly chosen nodes
        filename: file in which to store the graph
    """
    # Generate the network
    G = nx.erdos_renyi_graph(N, p)

    # Save to file
    nx.write_pajek(G, filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=5000, help='network size')
    parser.add_argument('-p', type=float, default=0.1, help='probability of connecting two random nodes')
    parser.add_argument('-f', type=str, help='filename for graph output')
    parser.add_argument('--ER', action='store_true', help='generate an ER random graph')
    args = parser.parse_args()

    if args.ER:
        gen_ER_model(args.N, args.p, args.f)