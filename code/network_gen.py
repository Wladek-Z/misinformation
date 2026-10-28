import networkx as nx
import argparse
import numpy as np
import graph_tool.all as gt
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

def generate_dcsbm_ba(N, K, m, mu):
    """
    Generates a DCSBM graph with a Barabási-Albert degree sequence using graph-tool,
    then converts to a NetworkX graph. Also performs a rewiring algorithm to remove
    multi-edges and self-edges while preserving group membership and degree sequence.
    
    Parameters
    ----------
    N: int
        number of nodes.
    K: int
        number of blocks/communities.
    m: str
        Barabási-Albert attachment parameter (average degree approx. 2m).
    mu: float
        mixing parameter (fraction of inter-block edges).
        
    Returns
    -------
    G: networkx Graph
        the generated degree-corrected SBM graph with group assignment as
        node attributes
    """
    # resolve filepath to corresponding BA network
    f = f'graphs/pajek/BA/0/{N} {m}.net.gz'
    # extract the Barabási-Albert degree sequence
    ba_graph = nx.Graph(nx.read_pajek(f))
    degs = np.array([d for _, d in ba_graph.degree()], dtype=np.int32)
    D = degs.sum()
    
    # assign nodes into K equal-sized groups
    # shuffle to ensure high-degree hub nodes are distributed evenly across blocks
    b = np.repeat(np.arange(K), int(np.ceil(N / K)))[:N]
    np.random.shuffle(b)
    
    # construct the expected edge-count matrix by assuming constant intrablock 
    # connections and interblock connections
    probs = np.full((K, K), (mu * D) / (K * (K - 1.0)))
    np.fill_diagonal(probs, ((1.0 - mu) * D) / K)
                
    # generate the graph using graph-tool's DCSBM generator
    g = gt.generate_sbm(
        b=b,
        probs=probs,
        out_degs=degs,
        directed=False,
        micro_degs=False,
        micro_ers=False
    )

    # convert numpy block array 'b' into a graph-tool Vertex Property Map
    b_prop = g.new_vertex_property("int")
    b_prop.a = b

    # run random_rewire with the correct parameter names
    swaps = gt.random_rewire(
        g,
        model='constrained-configuration',    # Preserves exact in/out-degree sequences
        n_iter=10,                            # Number of full passes over the edge list
        edge_sweep=True,                      # Iterates through edges sequentially
        block_membership=b_prop,              # Restricts edge swaps to matching block pairs
        self_loops=False,                     # Rejects proposed swaps that create self-loops
        parallel_edges=False                  # Rejects proposed swaps that create parallel edges
    )

    # convert to networkx graph
    G = graph_tool_to_networkx(g)
    nx.set_node_attributes(
        G,
        {node: int(b[node]) for node in G.nodes},
        name="block"
    )
    
    return G

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
        # generate ER network
        if model == 'ER':
            G = nx.erdos_renyi_graph(N, float(p))
        elif model == 'BA':
            # generate BA network on initial complete graph
            g = nx.complete_graph(int(p))
            G = nx.barabasi_albert_graph(N, int(p), initial_graph=g)

        # save to file
        output = f'./graphs/pajek/{model}/{i}'
        os.makedirs(output, exist_ok=True)
        nx.write_pajek(G, f'{output}/{N} {''.join(p.split('.'))}.net.gz')

def gen_dcsbm_samples(N, K, m, mu, n):
    """
    Generate samples of a DCSB model given 2 input parameters.
    
    Parameters
    ----------
        N: int
            network size
        K: int
            number of groups
        m: str
            input parameter point (1)
        mu: str
            input parameter point (2)
        n: str
            number of networks to produce for each parameters
    """
    # generate n graphs for each point in parameter space
    for i in range(n):
        # generate DCSBM network
        G = generate_dcsbm_ba(N, K, m, float(mu))
        # save graph to file
        output = f'./graphs/pajek/DCSB/{i}'
        os.makedirs(output, exist_ok=True)
        nx.write_pajek(G, f'{output}/{N} {m} {''.join(mu.split('.'))}.net.gz')

def graph_tool_to_networkx(g: gt.Graph) -> nx.Graph:
    """Convert a graph-tool graph to a NetworkX graph."""
    G = nx.Graph()
    G.add_nodes_from(range(g.num_vertices()))
    G.add_edges_from(
        (int(e.source()), int(e.target()))
        for e in g.edges()
    )
    return G

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=5000, help='network size')
    parser.add_argument('-n', type=int, default=1, help='number of graphs to produce per parameter point')
    parser.add_argument('-p', type=str, help='probability of connecting two random nodes')
    parser.add_argument('-f', type=str, help='file containing graph parameter input')
    parser.add_argument('--gen', action='store_true', help='generate a graph')
    parser.add_argument('--samples', action='store_true', help='generate graph samples for given parameter point')
    parser.add_argument('-m', type=str, choices=['ER', 'BA', 'DCSB'], help='Choose network model to generate')
    args = parser.parse_args()

    if args.gen:
        gen_model(args.N, args.p, args.m)
    elif args.samples:
        if args.m == 'DCSB':
            gen_dcsbm_samples()
        else:
            gen_graph_samples(args.N, args.p, args.n, args.m)