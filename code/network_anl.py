import argparse
import networkx as nx
import numpy as np
import pandas as pd
import os
from scipy.sparse.linalg import eigsh
from networkx.algorithms.community import modularity

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

def average_clustering(G):
    """
    Return the average clustering coefficient and its Fano factor.

    Parameters
    ----------
        G: NetworkX graph
            the network as a networkx graph

    Returns
    -------
        mean_clustering: float
            the mean node clustering coefficient of the graph
        fano_factor: float
            the variance of clustering coefficients divided by their mean
    """
    clustering_coefficients = np.fromiter(
        nx.clustering(G).values(),
        dtype=float
    )
    mean_clustering = (
        clustering_coefficients.mean()
        if clustering_coefficients.size
        else np.nan
    )
    fano_factor = (
        clustering_coefficients.var() / mean_clustering
        if clustering_coefficients.size and mean_clustering != 0
        else np.nan
    )
    return mean_clustering, fano_factor

def all_metrics(n, p, N, model, q=None, K=None):
    """
    Read in graphs for a given parameter point and perform analysis over all relevant
    global metrics. Save results to file.
    
    Parameters
    ----------
        n: int
            number of sampled graphs per parameter point
        p: str
            first input parameter point
        N: str
            network size
        model: str
            relevant network model
        q: str
            second input parameter point
        K: int
            number of groups
    """
    # initialise property arrays
    properties = [
        'avg_k',
        'avg_cc',
        'transitivity',
        'avg_path_length',
        'diameter',
        'k_assortativity',
        'n_components',
        'F_factor_k',
        'F_factor_cc',
        'a_connectivity',
        'spectral_r'
    ]
    # add modularity if working with DCSBM
    if model == 'DCSB':
        properties.append('modularity')

    # initialise data structure
    data = {property_name: [] for property_name in properties}

    # iterate over each graph
    for i in range(n):
        print(i)
        # resolve filepath to graph
        if model == 'DCSB':
            f = f'./graphs/pajek/{model}/{i}/{N} {K} {''.join(p.split('.'))} {''.join(q.split('.'))}.net.gz'
        elif model == 'SWSF':
            f = f'./graphs/pajek/{model}/{i}/{N} {''.join(p.split('.'))} {''.join(q.split('.'))}.net.gz'    
        else:
            f = f'./graphs/pajek/{model}/{i}/{N} {''.join(p.split('.'))}.net.gz'

        # load in graph
        G = nx.Graph(nx.read_pajek(f))

        # calculate number of connected components
        ncc_current = nx.number_connected_components(G)
        # calculate average degree and Fano factor
        avg_k, fano_factor_k = average_degree(G)
        avg_cc, fano_factor_cc = average_clustering(G)
        # calculate spectral radius from adjacency matrix
        A = nx.to_scipy_sparse_array(G, dtype=float)
        spectral_radius = eigsh(
            A,
            k=1,
            which='LM',
            return_eigenvectors=False
        )[0]
        # calculate algebraic connectivity from Laplacian matrix
        L = nx.laplacian_matrix(G).astype(float)
        eigenvalues = eigsh(
            L,
            k=2,
            which='SA',
            return_eigenvectors=False
        )
        a_connectivity = np.sort(eigenvalues)[1]
        # calculate remaining metrics
        values = {
            'avg_k': avg_k,
            'avg_cc': avg_cc,
            'transitivity': nx.transitivity(G),
            'k_assortativity': nx.degree_assortativity_coefficient(G),
            'n_components': ncc_current,
            'F_factor_k': fano_factor_k,
            'F_factor_cc': fano_factor_cc,
            'a_connectivity': a_connectivity,
            'spectral_r': spectral_radius
        }

        # Leave path-related metrics empty for disconnected graphs.
        if ncc_current == 1:
            values['avg_path_length'] = nx.average_shortest_path_length(G)
            values['diameter'] = nx.diameter(G)
        else:
            values['avg_path_length'] = None
            values['diameter'] = None

        # calculate modularity for DCSBM only
        if model == 'DCSB':
            communities_by_block = {block_id: set() for block_id in range(K)}
            for node, node_data in G.nodes(data=True):
                if "block" not in node_data:
                    raise ValueError(
                        f"DCSBM graph {f} is missing the 'block' node attribute"
                    )
                block_id = int(node_data["block"])
                if block_id not in communities_by_block:
                    raise ValueError(
                        f"DCSBM graph {f} contains invalid block {block_id}; "
                        f"expected values from 0 to {K - 1}"
                    )
                communities_by_block[block_id].add(node)
            communities = [
                communities_by_block[block_id]
                for block_id in range(K)
            ]
            values['modularity'] = modularity(G, communities)

        for property_name in properties:
            data[property_name].append(values[property_name])

    # construct dataframe
    df = pd.DataFrame(data)

    # save to file
    output = f'graphs/data/{model}/uncombined'
    os.makedirs(output, exist_ok=True)
    if model == 'DCSB':
        df.to_csv(f'{output}/{N} {K} {''.join(p.split('.'))} {''.join(q.split('.'))}.txt')
    elif model == 'SWSF':
        df.to_csv(f'{output}/{N} {''.join(p.split('.'))} {''.join(q.split('.'))}.txt')
    else:
        df.to_csv(f'{output}/{N} {''.join(p.split('.'))}.txt')

def analyse_means(param_file, model, N):
    """
    Combine the statistics for the individually sampled graphs to obtain
    mean global properties with errors.

    Parameters
    ----------
        param_file: str
            path to file containing input parameters
        model: str
            network model to analyse
        N: str
            network size
    """
    # read in parameters
    p_values = np.loadtxt(param_file, dtype=str)
    p_values = np.atleast_1d(p_values)

    # initialise property arrays
    properties = [
        'avg_k',
        'avg_cc',
        'transitivity',
        'avg_path_length',
        'diameter',
        'k_assortativity',
        'n_components',
        'F_factor_k',
        'F_factor_cc',
        'a_connectivity',
        'spectral_r'
    ]

    # initialise data structure
    data = {'p': []}

    for property_name in properties:
        data[f'{property_name}_mean'] = []
        data[f'{property_name}_sem'] = []

    # iterate over each parameter point
    for p in p_values:
        # print current parameter value
        print(p)

        # resolve filepath to graph
        f = f'./graphs/data/{model}/uncombined/{N} {''.join(p.split('.'))}.txt'

        # read in data
        df = pd.read_csv(f)

        # add parameter value to data
        data['p'].append(float(p))

        # iterate over each measured property
        for property_name in properties:
            # set missing elements to NaNs
            values = pd.to_numeric(df[property_name], errors='coerce')

            # a missing sample makes this metric unavailable for this p value
            if values.isna().any():
                data[f'{property_name}_mean'].append(None)
                data[f'{property_name}_sem'].append(None)
                continue

            data[f'{property_name}_mean'].append(values.mean())
            data[f'{property_name}_sem'].append(values.sem())

    # construct dataframe
    df = pd.DataFrame(data)

    # save to file
    output = f'./graphs/data/{model}/combined'
    os.makedirs(output, exist_ok=True)
    df.to_csv(f'{output}/{N}.txt', index=False)

def analyse_means_2d(param_file_1, param_file_2, model, N, K=None):
    """
    Combine the statistics for the individually sampled graphs in a 2-dimensional
    parameter space to obtain mean global properties with errors.

    Parameters
    ----------
        param_file_1: str
            path to file containing primary input parameters
        param_file_1: str
            path to file containing secondary input parameters
        model: str
            network model to analyse
        N: str
            network size
        K: int
            number of communities, if applicable
    """
    # read in parameters
    p_values = np.loadtxt(param_file_1, dtype=str)
    p_values = np.unique(np.atleast_1d(p_values))
    q_values = np.loadtxt(param_file_2, dtype=str)
    q_values = np.unique(np.atleast_1d(q_values))

    # initialise property arrays
    properties = [
        'avg_k',
        'avg_cc',
        'transitivity',
        'avg_path_length',
        'diameter',
        'k_assortativity',
        'n_components',
        'F_factor_k',
        'F_factor_cc',
        'a_connectivity',
        'spectral_r'
    ]
    # add modularity for DCSBM
    if model == 'DCSB':
        properties.append('modularity')

    # initialise data structure
    data = {
        'p': [],
        'q': []
    }

    # add data columns for each property
    for property_name in properties:
        data[f'{property_name}_mean'] = []
        data[f'{property_name}_sem'] = []

    # iterate over each parameter point in 2 dimensions
    for p in p_values:
        for q in q_values:
            # print current parameter point
            print(p, q)

            # resolve filepath to graph
            if model == 'DCSB':
                f = f'./graphs/data/{model}/uncombined/{N} {K} {''.join(p.split('.'))} {''.join(q.split('.'))}.txt'
            elif model == 'SWSF':
                f = f'./graphs/data/{model}/uncombined/{N} {''.join(p.split('.'))} {''.join(q.split('.'))}.txt'
            
            # read in data
            df = pd.read_csv(f)

            # add parameter values to data
            data['p'].append(float(p))
            data['q'].append(float(q))

            # iterate over each measured property
            for property_name in properties:
                # set missing elements to NaNs
                values = pd.to_numeric(df[property_name], errors='coerce')

                # a missing sample makes this metric unavailable for this p value
                if values.isna().any():
                    data[f'{property_name}_mean'].append(None)
                    data[f'{property_name}_sem'].append(None)
                    continue

                data[f'{property_name}_mean'].append(values.mean())
                data[f'{property_name}_sem'].append(values.sem())

    # construct dataframe
    df = pd.DataFrame(data)

    # save to file
    output = f'./graphs/data/{model}/combined'
    os.makedirs(output, exist_ok=True)
    if model == 'DCSB':
        df.to_csv(f'{output}/{N} {K}.txt', index=False)
    elif model == 'SWSF':
        df.to_csv(f'{output}/{N}.txt', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='filename for graph input')
    parser.add_argument('-out', type=str, help='filename for analysis output')
    parser.add_argument('-m', type=str, choices=['ER', 'BA', 'DCSB', 'SWSF'], help='Choose network model to analyse')
    parser.add_argument('-p', type=str, help='primary input parameter value/file')
    parser.add_argument('-q', type=str, default=None, help='secondary input parameter value/file')
    parser.add_argument('-K', type=int, default=None, help='number of groups')
    parser.add_argument('-N', type=str, default='5000', help='network size')
    parser.add_argument('-n', type=int, default=10, help='number of graphs per parameter point')
    parser.add_argument('--avgdeg', action='store_true', help='output the average degree of a graph')
    parser.add_argument('--avgcc', action='store_true', help='output the average clustering coefficient of a graph')
    parser.add_argument('--all', action='store_true', help='analyse all properties for an input parameter point')
    parser.add_argument('--mean', action='store_true', help='combine sampled graph statistics')
    parser.add_argument('--mean2d', action='store_true', help='combine sampled graph statistics in 2D')
    args = parser.parse_args()

    if args.avgdeg:
        G = nx.Graph(nx.read_pajek(args.f))
        average, fano_factor = average_degree(G)
        print(f'average degree: {average}')
        print(f'Fano factor: {fano_factor}')
    elif args.avgcc:
        G = nx.Graph(nx.read_pajek(args.f))
        average, fano_factor = average_clustering(G)
        print(f'average clustering coefficient: {average}')
        print(f'Fano factor: {fano_factor}')
    elif args.all:
        all_metrics(args.n, args.p, args.N, args.m, args.q, args.K)
    elif args.mean:
        analyse_means(args.p, args.m, args.N)
    elif args.mean2d:
        analyse_means_2d(args.p, args.q, args.m, args.N, args.K)
