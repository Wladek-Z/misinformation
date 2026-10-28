import numpy as np
import graph_tool.all as gt
import networkx as nx

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
    
    # assign nodes into K equal-sized groups
    # shuffle to ensure high-degree hub nodes are distributed evenly across blocks
    b = np.repeat(np.arange(K), int(np.ceil(N / K)))[:N]
    np.random.shuffle(b)
    
    # calculate actual total degree volume D_r for each block
    D_r = np.array([degs[b == r].sum() for r in range(K)], dtype=np.float64)
    
    # construct the expected edge-count matrix `probs` (E_rs)
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

# Example usage:
g, block_assignments = generate_dcsbm_ba(N=5000, K=20, m=50, mu=0.3)
print(f"Generated DCSBM graph with {g.num_vertices()} nodes and {g.num_edges()} edges.")