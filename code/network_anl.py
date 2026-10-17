import argparse
import networkx as nx

def average_degree(filename):
    """
    Print the average degree of a given graph.
    
    Arguments:
        filename: filepath to stored graph data
    """
    # read in the graph
    G = nx.read_pajek(filename)

    # Calculate and output average degree
    avg_k = sum([d for (n, d) in nx.degree(G)]) / float(G.number_of_nodes())

    print(avg_k)

def analyse_all(N, param_file, graph_model):
    """
    Read in each graph from a parameter file and perform analysis over all
    relevant global metrics. Save results to file.
    
    Arguments:
        N: network size
        param_file: file containing graph parameters
        graph_model: network model to analyse
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='filename for graph input')
    parser.add_argument('-out', type=str, help='filename for analysis output')
    parser.add_argument('-m', type=str, choices=['ER'], help='Choose network model to analyse')
    parser.add_argument('-p', type=str, help='file containing parameter list(s)')
    parser.add_argument('--avgdeg', action='store_true', help='Output the average degree of a graph')
    args = parser.parse_args()

    if args.avgdeg:
        average_degree(args.f)