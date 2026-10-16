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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='filename for graph output')
    parser.add_argument('--avgdeg', action='store_true', help='Output the average degree of a graph')
    args = parser.parse_args()

    if args.avgdeg:
        average_degree(args.f)