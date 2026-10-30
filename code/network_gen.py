import networkx as nx
import argparse
import numpy as np
import graph_tool.all as gt
import os
import random

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

def gen_model_2d(N, p, q, model, K=None):
    """
    Generate a network using a double-parameter graph model.
    
    Parameters
    ----------
        N: int
            network size
        p: str
            first input parameter
        q: str
            second input parameter
        model: str
            which graph model to use
        K: int
            number of groups, if applicable
    """
    # choose graph model
    if model == 'DCSB':
        G = generate_dcsbm_ba(N, K, p, float(q))
        nx.write_pajek(G, f'./graphs/pajek/{model}/{N} {K} {''.join(p.split('.'))} {''.join(q.split('.'))}.net.gz')
    elif model == 'SWSF':
        G = generate_SWSF(N, p, q)
        nx.write_pajek(G, f'./graphs/pajek/{model}/{N} {''.join(p.split('.'))} {''.join(q.split('.'))}.net.gz')

def generate_swsf(N, m, mu):
    """
    Generate a Klemm-Eguiluz Small-World-Scale-Free model.

    Parameters
    ----------
        N: int
            network size
        m: int
            size of active nodes
        mu: float
            probability of edge connecting to non-active node

    Returns
    -------
        G: NetworkX Graph
            the generated SWSF network
    """
    # validate inputs
    if not isinstance(N, int) or isinstance(N, bool) or N < 1:
        raise ValueError("N must be a positive integer")
    if not isinstance(m, int) or isinstance(m, bool) or not 1 <= m <= N:
        raise ValueError("m must be an integer between 1 and N")
    if not isinstance(mu, (int, float)) or isinstance(mu, bool) or not 0 <= mu <= 1:
        raise ValueError("mu must be a number between 0 and 1")

    rng = random.Random()
    G = nx.complete_graph(m)
    active = list(G.nodes)

    # iterate over new nodes
    for new_node in range(m, N):
        deactivated = list(set(G.nodes) - set(active))
        targets = []

        # iterate over active nodes
        for active_node in active:
            # connect to active node with p = (1 - mu) or if there are 0 deactivated nodes
            if not deactivated or rng.random() >= mu:
                target = active_node
            else:
                connected = False
                while not connected:
                    # choose target among deactivated nodes
                    target = rng.choice(deactivated)
                    total_degree = sum(G.degree(node) for node in deactivated)
                    # connect with probability k_target / (sum degree of deactivated nodes)
                    connected = (
                        total_degree > 0
                        and rng.random() < G.degree(target) / total_degree
                    )
            targets.append(target)

        # add reciprocal edge from new node to target nodes
        G.add_node(new_node)
        G.add_edges_from((new_node, target) for target in targets)

        # add new node to active nodes
        active.append(new_node)
        chosen = False
        # iterate while candidate node not yet chosen
        while not chosen:
            # choose random candidate from active nodes
            candidate = rng.choice(active)
            # calculate deactivation probability
            inverse_degree_sum = sum(
                1 / G.degree(node) for node in active if G.degree(node) > 0
            )
            pd = (1 / G.degree(candidate)) / inverse_degree_sum
            # remove candidate from active nodes with p = p_d
            if rng.random() < pd:
                active.remove(candidate)
                chosen = True

    return G

def generate_dcsbm_ba(N, K, m, nu):
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
    nu: float
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
    probs = np.full((K, K), (nu * D) / (K * (K - 1.0)))
    np.fill_diagonal(probs, ((1.0 - nu) * D) / K)
                
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

def gen_swsfm_samples(N, m, mu, n):
    """
    Generate samples of a SWSF model given 2 input parameters.
    
    Parameters
    ----------
        N: int
            network size
        m: str
            first input parameter
        mu: str
            second input parameter
        n: str
            number of networks to produce for each parameters
    """
    # generate n graphs for each point in parameter space
    for i in range(n):
        # generate DCSBM network
        G = generate_swsf(N, int(m), float(mu))
        # save graph to file
        output = f'./graphs/pajek/SWSF/{i}'
        os.makedirs(output, exist_ok=True)
        nx.write_pajek(G, f'{output}/{N} {m} {''.join(mu.split('.'))}.net.gz')

def gen_dcsbm_samples(N, K, m, nu, n):
    """
    Generate samples of a DCSB model given 2 input parameters.
    
    Parameters
    ----------
        N: int
            network size
        K: int
            number of groups
        m: str
            first input parameter
        nu: str
            second input parameter
        n: str
            number of networks to produce for each parameters
    """
    # generate n graphs for each point in parameter space
    for i in range(n):
        # generate DCSBM network
        G = generate_dcsbm_ba(N, K, m, float(nu))
        # save graph to file
        output = f'./graphs/pajek/DCSB/{i}'
        os.makedirs(output, exist_ok=True)
        nx.write_pajek(G, f'{output}/{N} {K} {m} {''.join(nu.split('.'))}.net.gz')

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
    parser.add_argument('-n', type=int, default=10, help='number of graphs to produce per parameter point')
    parser.add_argument('-p', type=str, help='first graph input parameter')
    parser.add_argument('-q', type=str, help='second graph input parameter')
    parser.add_argument('-K', type=int, default=10, help='number of groups for DCSBM')
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