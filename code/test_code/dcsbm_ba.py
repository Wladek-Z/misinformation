import numpy as np
import graph_tool.all as gt
import networkx as nx
from networkx.algorithms.community import modularity

def generate_dcsbm_ba(N=5000, K=4, m=5, mu=0.2):
    """
    Generates a DCSBM graph with a Barabási-Albert degree sequence.
    
    Parameters
    ----------
    N: int
        number of nodes.
    K: int
        number of blocks/communities.
    m: int
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
    # generate the Barabási-Albert degree sequence
    ba_graph = nx.barabasi_albert_graph(N, m)
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
        n_iter=10,                # Number of full passes over the edge list
        edge_sweep=True,          # Iterates through edges sequentially
        block_membership=b_prop,  # Restricts edge swaps to matching block pairs
        self_loops=False,         # Rejects proposed swaps that create self-loops
        parallel_edges=False      # Rejects proposed swaps that create parallel edges
    )
    
    return g, b

def graph_tool_to_networkx(g: gt.Graph) -> nx.Graph:
    """Convert a graph-tool graph to a NetworkX graph."""
    G = nx.Graph()
    G.add_nodes_from(range(g.num_vertices()))
    G.add_edges_from(
        (int(e.source()), int(e.target()))
        for e in g.edges()
    )
    return G

def average_degree(G):
    """
    Return the average degree and Fano factor of a given graph.

    Parameters
    ----------
        G: NetworkX graph
            the network as a networkx graph

    Returns
    -------
        mean_degree: float
            the mean node degree of the graph
        fano_factor: float
            the variance of degree divided by the mean degree of the graph
    """
    degrees = np.fromiter((degree for _, degree in nx.degree(G)), dtype=float)
    mean_degree = degrees.mean() if degrees.size else np.nan
    fano_factor = (
        degrees.var() / mean_degree
        if degrees.size and mean_degree != 0
        else np.nan
    )
    return mean_degree, fano_factor

if __name__ == "__main__":

    # Example usage:
    g, b = generate_dcsbm_ba(N=5000, K=10, m=50, mu=0.01)
    print(f"Generated DCSBM graph with {g.num_vertices()} nodes and {g.num_edges()} edges.")

    G = graph_tool_to_networkx(g)
    nx.set_node_attributes(
        G,
        {node: int(b[node]) for node in G.nodes},
        name="block"
    )

    mean_k, _ = average_degree(G)
    print(f"Average degree: {mean_k}")

    communities = [
        {node for node, data in G.nodes(data=True)
        if data["block"] == block_id}
        for block_id in sorted(set(b))
    ]

    Q = modularity(G, communities)
    print(f"Modularity: {Q}")
