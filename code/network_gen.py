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
    Generates a DCSBM graph with a Barabási-Albert degree sequence.
    
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
    g: graph_tool Graph
        the generated degree-corrected SBM graph.
    b: numpy ndarray
        array of length N containing block assignments (0 to K-1).
    """
    # resolve filepath to corresponding BA network
    f = f'graphs/pajek/BA/0/{N} {m}.net.gz'
    # extract the Barabási-Albert degree sequence
    ba_graph = nx.Graph(nx.read_pajek(f))
    degs = np.array([d for _, d in ba_graph.degree()], dtype=np.int32)
    
    # assign nodes into K equal-sized groups
    # shuffle to ensure high-degree hub nodes are distributed evenly across blocks
    b = np.repeat(np.arange(K), int(np.ceil(N / K)))[:N]
    np.random.shuffle(b)
    
    # calculate actual total degree volume D_r for each block
    D_r = np.array([degs[b == r].sum() for r in range(K)], dtype=np.float64)
    
    # construct the expected edge-count matrix probs (E_rs)
    # for undirected graphs in graph-tool:
    # - probs[r, r] is internal edges in block r: (1 - mu) * D_r / 2
    # - probs[r, s] is external edges between r and s: mu * D_r / (K - 1)
    probs = np.zeros((K, K), dtype=np.float64)
    for r in range(K):
        for s in range(K):
            if r == s:
                probs[r, r] = (1.0 - mu) * D_r[r] / 2.0
            else:
                probs[r, s] = (mu * D_r[r]) / (K - 1.0)
                
    # generate the graph using graph-tool's DCSBM generator
    g = gt.generate_sbm(
        b=b,
        probs=probs,
        out_degs=degs,
        directed=False
    )
    
    return g, b

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
        g, b = generate_dcsbm_ba(N, K, m, float(mu))
        # convert graph-tool to networkx
        G = graph_tool_to_networkx(g)
        # set block assignments as node attributes
        nx.set_node_attributes(
            G,
            {node: int(b[node]) for node in G.nodes},
            name="block"
        )
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