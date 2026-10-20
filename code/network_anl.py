import argparse
import networkx as nx
import numpy as np
import pandas as pd

def average_degree(G):
    """
    Return the average degree of a given graph.
    
    Arguments:
        G: the network as a networkx graph

    Returns:
        avg_k: the average degree of the graph
    """
    # calculate and output average degree
    avg_k = sum([d for (n, d) in nx.degree(G)]) / float(G.number_of_nodes())

    return avg_k

def analyse_all(param_file, output, N, model):
    """
    Read in each graph from a parameter file and perform analysis over all
    relevant global metrics. Save results to file.
    
    Arguments:
        param_file: file containing graph parameters
        output: location to save analysis results
        N: network size
        model: relevant network model
    """
    # read in parameters
    params = np.loadtxt(param_file, dtype=str)

    # initialise metric result arrays
    deg = []    # average degree
    acc = []    # average clustering coefficient
    gcc = []    # transitivity
    apl = []    # average path length
    ndm = []    # diameter
    ass = []    # degree assortativity
    ncc = []    # number of connected components

    # iterate over each parameter
    for p in params:
        # resolve filepath to graph
        f = f'./graphs/pajek/{model}/{N} {''.join(p.split('.'))}.net.gz'

        # load in graph
        G = nx.read_pajek(f)

        # calculate metrics
        deg.append(average_degree(G))
        acc.append(nx.average_clustering(G))
        gcc.append(nx.transitivity(G))
        ass.append(nx.degree_assortativity_coefficient(G))
        ncc_current = nx.number_connected_components(G)
        ncc.append(ncc_current)

        # check for disconnected components
        if ncc_current == 1:
            apl.append(nx.average_shortest_path_length(G))
            ndm.append(nx.diameter(G))
        else:
            apl.append(None)
            ndm.append(None)

    # construct dataframe
    data = {
        'p': params,
        'avg_k': deg,
        'avg_cc': acc,
        'transitivity': gcc,
        'avg_path_length': apl,
        'diameter': ndm,
        'k_assortativity': ass,
        'n_components': ncc
    }
    df = pd.DataFrame(data)
    df.to_csv(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='filename for graph input')
    parser.add_argument('-out', type=str, help='filename for analysis output')
    parser.add_argument('-m', type=str, choices=['ER'], help='Choose network model to analyse')
    parser.add_argument('-p', type=str, help='file containing parameter list(s)')
    parser.add_argument('-N', type=str, default='5000', help='network size')
    parser.add_argument('--avgdeg', action='store_true', help='Output the average degree of a graph')
    args = parser.parse_args()

    if args.avgdeg:
        G = nx.read_pajek(args.f)
        print(average_degree(G))