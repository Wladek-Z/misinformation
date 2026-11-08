import networkx as nx
import random

def gen_SWSF(N, m, mu):
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
