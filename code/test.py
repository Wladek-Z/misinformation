from time import perf_counter
import networkx as nx
from scipy.sparse.linalg import eigsh

G = nx.erdos_renyi_graph(5000, 0.02)
start = perf_counter()
adjacency = nx.to_scipy_sparse_array(G, dtype=float)
spectral_radius = eigsh(
    adjacency,
    k=1,
    which='LM',
    return_eigenvectors=False
)[0]
print(f"spectral radius time: {perf_counter() - start}")
start = perf_counter()

a_connectivity = nx.algebraic_connectivity(G)

print(f"algebraic connectivity time: {perf_counter() - start}")

