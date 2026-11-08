import networkx as nx
from networkx.algorithms.community import modularity


def graph_tool_to_networkx(g):
    G = nx.Graph()
    G.add_nodes_from(range(g.num_vertices()))
    G.add_edges_from(
        (int(e.source()), int(e.target()))
        for e in g.edges()
    )
    return G


g, b = generate_dcsbm_ba(N, K, m, float(mu))

G = graph_tool_to_networkx(g)

nx.set_node_attributes(
    G,
    {node: int(b[node]) for node in G.nodes},
    name="block"
)

communities = [
    {node for node, data in G.nodes(data=True)
     if data["block"] == block_id}
    for block_id in sorted(set(b))
]

Q = modularity(G, communities)