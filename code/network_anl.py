import argparse
import networkx as nx
import numpy as np
import pandas as pd
import os
from scipy.sparse.linalg import eigsh

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

def all_metrics(n, p, N, model):
    """
    Read in graphs for a single parameter point and perform analysis over all relevant
    global metrics. Save results to file.
    
    Parameters
    ----------
        n: int
            number of sampled graphs per parameter point
        p: str
            input parameter point
        N: str
            network size
        model: str
            relevant network model
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
        'F_factor',
        'a_connectivity',
        'spectral_r'
    ]

    # initialise data structure
    data = {property_name: [] for property_name in properties}

    # iterate over each graph
    for i in range(n):
        print(i)
        # resolve filepath to graph
        f = f'./graphs/pajek/{model}/{i}/{N} {''.join(p.split('.'))}.net.gz'

        # load in graph
        G = nx.Graph(nx.read_pajek(f))

        # calculate number of connected components
        ncc_current = nx.number_connected_components(G)
        # calculate average degree and Fano factor
        avg_k, fano_factor = average_degree(G)
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
            'avg_cc': nx.average_clustering(G),
            'transitivity': nx.transitivity(G),
            'k_assortativity': nx.degree_assortativity_coefficient(G),
            'n_components': ncc_current,
            'F_factor': fano_factor,
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

        for property_name in properties:
            data[property_name].append(values[property_name])

    # construct dataframe
    df = pd.DataFrame(data)

    # save to file
    output = f'graphs/data/{model}/uncombined'
    os.makedirs(output, exist_ok=True)
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
        'F_factor',
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='filename for graph input')
    parser.add_argument('-out', type=str, help='filename for analysis output')
    parser.add_argument('-m', type=str, choices=['ER', 'BA'], help='Choose network model to analyse')
    parser.add_argument('-p', type=str, help='input parameter value/file')
    parser.add_argument('-N', type=str, default='5000', help='network size')
    parser.add_argument('-n', type=int, default=1, help='number of graphs per parameter point')
    parser.add_argument('--avgdeg', action='store_true', help='output the average degree of a graph')
    parser.add_argument('--all', action='store_true', help='analyse all properties for an input parameter point')
    parser.add_argument('--mean', action='store_true', help='combine sampled graph statistics')
    args = parser.parse_args()

    if args.avgdeg:
        G = nx.Graph(nx.read_pajek(args.f))
        average, fano_factor = average_degree(G)
        print(f'average degree: {average}')
        print(f'Fano factor: {fano_factor}')
    elif args.all:
        all_metrics(args.n, args.p, args.N, args.m)
    elif args.mean:
        analyse_means(args.p, args.m, args.N)
