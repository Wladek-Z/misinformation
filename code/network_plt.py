from matplotlib import pyplot as plt
import scienceplots
import pandas as pd
import argparse
import os
import numpy as np

plt.style.use('science')
plt.rcParams['text.usetex'] = False

def plot_y_vs_x_with_errors(x, y, y_err, xlabel, ylabel):
    """
    Plot some global network metric against input parameter with associated
    y-axis errors.
    
    Parameters
    ----------
        x: ndarray
            the independent input parameters
        y: ndarray
            the dependent global metric
        y_err: ndarray
            the standard errors on the means of the global metrics
        xlabel: str
            the x-axis label
        ylabel: str
            the y-axis label
    
    Returns
    -------
        fig: matplotlib Figure
            the generated figure
        ax: matplotlib Axes
            the axis of the figure
    """
    fig, ax = plt.subplots(1, 1, layout='constrained', figsize=(8, 6))

    ax.errorbar(x, y, yerr=y_err, fmt='o-b', label='numerical')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.5)

    return fig, ax

def plot_all_1d(model, N, n):
    """
    Plot all global metrics against input parameter for a given network model
    at a given network size. Function for network models with a 1-dimensional
    parameter space.
    
    Parameters
    ----------
        model: str
            which network model to plot results for
        N: int
            size of network model used in results
        n: int
            number of graphs used to calculate mean
    """
    # load in network data
    df = pd.read_csv(f'graphs/data/{model}/combined/{N}.txt')

    # set x-labels based on network model
    if model == 'ER':
        xlabel = 'edge probability, $p$'
    elif model == 'BA':
        xlabel = 'edges per new node, $m$'

    ylabels = [
        r'average degree, $\langle k \rangle$',
        r'average clustering coefficient, $\langle C \rangle$',
        'transitivity, $T$',
        'average path length, $l$',
        'diameter, $d$',
        'degree assortativity, $r$',
        '# connected components, $n_c$',
        'degree Fano factor, $F$',
        'algebraic connectivity $a$',
        r'spectral radius, $\rho$'
    ]

    # iterate over each metric in the dataframe
    for metric, error, ylabel in zip(df.columns[1::2], df.columns[2::2], ylabels):
        # plot figure
        fig, ax = plot_y_vs_x_with_errors(df['p'], df[metric], df[error], xlabel, ylabel)

        # add analytical expressions for certain models and metrics
        if model == 'ER':
            if metric == 'avg_k_mean':
                y_analytical = df['p'] * N
                ax.plot(df['p'], y_analytical, color='red', linestyle='--', label='analytical')
                ax.legend(loc='upper center')
            elif metric == 'avg_cc_mean':
                y_analytical = df['p']
                ax.plot(df['p'], y_analytical, color='red', linestyle='--', label='analytical')
                ax.legend(loc='upper center')
        elif model == 'BA' and metric == 'avg_k_mean':
            y_analytical = (2 * N - df['p'] - 1) * df['p'] / N
            ax.plot(df['p'], y_analytical, color='red', linestyle='--', label='analytical')
            ax.legend(loc='upper center')

        # define filepath and save figure
        filename = f'./graphs/plots/{model}'
        os.makedirs(filename, exist_ok=True)
        plt.savefig(f'{filename}/{metric} {N}')

    # also plot standard deviation of degree assortativity
    k_ass_std = df['k_assortativity_sem'] * np.sqrt(n)
    ylabel = r'standard deviation of $r$, $\sigma_r$'
    fig, ax = plot_y_vs_x_with_errors(df['p'], k_ass_std, 0, xlabel, ylabel)
    # define filepath and save figure
    filename = f'./graphs/plots/{model}'
    os.makedirs(filename, exist_ok=True)
    plt.savefig(f'{filename}/{'k_assortativity_std'} {N}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=5000, help='network size')
    parser.add_argument('--all', action='store_true', help='plot all metrics for a 1D parameter space')
    parser.add_argument('-m', type=str, choices=['ER', 'BA'], help='Choose network model to generate')
    parser.add_argument('-n', type=int, default=1, help='number of graphs per parameter point')
    args = parser.parse_args()

    if args.all:
        plot_all_1d(args.m, args.N, args.n)