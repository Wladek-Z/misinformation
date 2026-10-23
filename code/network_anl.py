import argparse
import networkx as nx
import numpy as np
import pandas as pd
import os

def average_degree(G):
    """
    Return the average degree of a given graph.

    Parameters
    ----------
        G: NetworkX graph
            the network as a networkx graph

    Returns
    -------
        float
            the average degree of the graph
    """
    # calculate and output average degree
    return sum([d for (n, d) in nx.degree(G)]) / float(G.number_of_nodes())

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
        'transitivity'
        'avg_path_length',
        'diameter',
        'k_assortativity',
        'n_components',
    ]

    # initialise data structure
    data = {property_name: [] for property_name in properties}

    # iterate over each graph
    for i in range(n):
        # resolve filepath to graph
        f = f'./graphs/pajek/{model}/{i}/{N} {''.join(p.split('.'))}.net.gz'

        # load in graph
        G = nx.Graph(nx.read_pajek(f))

        # calculate metrics
        ncc_current = nx.number_connected_components(G)
        values = {
            'avg_k': average_degree(G),
            'avg_cc': nx.average_clustering(G),
            'transitivity': nx.transitivity(G),
            'k_assortativity': nx.degree_assortativity_coefficient(G),
            'n_components': ncc_current,
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

def add_connectivity(param_file, model, N):
    """
    Calculate node connectivity for every sampled graph and add it to the
    existing uncombined and combined results.

    Parameters
    ----------
        param_file: str
            path to file containing input parameters
        model: str
            network model to analyse
        N: str
            network size
    """
    p_values = np.loadtxt(param_file, dtype=str)
    p_values = np.atleast_1d(p_values)
    graph_root = f'./graphs/pajek/{model}'
    uncombined_root = f'./graphs/data/{model}/uncombined'
    combined_file = f'./graphs/data/{model}/combined/{N}.txt'

    connectivity_means = []
    connectivity_sems = []

    for p in p_values:
        parameter = ''.join(p.split('.'))
        graph_paths = []

        for sample in sorted(os.listdir(graph_root), key=lambda name: int(name) if name.isdigit() else -1):
            sample_path = os.path.join(graph_root, sample)
            if not sample.isdigit() or not os.path.isdir(sample_path):
                continue

            graph_path = os.path.join(sample_path, f'{N} {parameter}.net.gz')
            if os.path.isfile(graph_path):
                graph_paths.append(graph_path)

        if not graph_paths:
            raise FileNotFoundError(
                f'No sampled graphs found for parameter {p} in {graph_root}'
            )

        connectivity = []
        for graph_path in graph_paths:
            graph = nx.Graph(nx.read_pajek(graph_path))
            connectivity.append(nx.node_connectivity(graph))

        uncombined_file = os.path.join(
            uncombined_root, f'{N} {parameter}.txt'
        )
        if not os.path.isfile(uncombined_file):
            raise FileNotFoundError(
                f'Uncombined data file not found: {uncombined_file}'
            )

        uncombined = pd.read_csv(uncombined_file)
        if len(uncombined) != len(connectivity):
            raise ValueError(
                f'{uncombined_file} has {len(uncombined)} rows, but '
                f'{len(connectivity)} sampled graphs were found'
            )

        uncombined['connectivity'] = connectivity
        uncombined.to_csv(uncombined_file, index=False)

        values = pd.Series(connectivity, dtype=float)
        connectivity_means.append(values.mean())
        connectivity_sems.append(values.sem())

    if not os.path.isfile(combined_file):
        raise FileNotFoundError(f'Combined data file not found: {combined_file}')

    combined = pd.read_csv(combined_file)
    if len(combined) != len(p_values):
        raise ValueError(
            f'{combined_file} has {len(combined)} rows, but '
            f'{len(p_values)} parameters were supplied'
        )

    for column in ('connectivity_mean', 'connectivity_sem'):
        if column in combined:
            combined = combined.drop(columns=column)

    combined['connectivity_mean'] = connectivity_means
    combined['connectivity_sem'] = connectivity_sems
    combined.to_csv(combined_file, index=False)


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
    parser.add_argument('--connectivity', action='store_true', help='add node connectivity to saved statistics')
    args = parser.parse_args()

    if args.avgdeg:
        G = nx.Graph(nx.read_pajek(args.f))
        print(average_degree(G))
    elif args.all:
        all_metrics(args.n, args.p, args.N, args.m)
    elif args.mean:
        analyse_means(args.p, args.m, args.N)
    elif args.connectivity:
        add_connectivity(args.p, args.m, args.N)
