from time import perf_counter
import networkx as nx
from scipy.sparse.linalg import eigsh
import numpy as np

g = nx.complete_graph(50)
G = nx.barabasi_albert_graph(5000, 50, initial_graph=g)
start = perf_counter()


laplacian = nx.laplacian_matrix(G).astype(float)

eigenvalues = eigsh(
    laplacian,
    k=2,
    which='SA',
    return_eigenvectors=False
)

a_connectivity = np.sort(eigenvalues)[1]

print(f"algebraic connectivity time: {perf_counter() - start}")
print(f"algebraic connectivity = {a_connectivity}")


start = perf_counter()

a_con = nx.algebraic_connectivity(G)

print(f"nx algebraic connectivity time: {perf_counter() - start}")