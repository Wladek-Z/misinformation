from matplotlib import pyplot as plt
import pandas as pd
import argparse
import os
import numpy as np
#import scienceplots

#plt.style.use('science')
#plt.rcParams['text.usetex'] = False

from pathlib import Path

STYLE_FILE = Path(__file__).with_name("shendrukGroupStyle.mplstyle")
plt.style.use(str(STYLE_FILE))

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
    fig, ax = plt.subplots(1, 1, layout='constrained')

    ax.errorbar(x, y, yerr=y_err, marker='o', linestyle='-', color=[172./255., 0, 64./255.], label='numerical')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.5)

    return fig, ax

def plot_2_in_1_y_vs_x_with_errors(x1, y1, x_err1, y_err1, x2, y2, x_err2, y_err2, xlabel, ylabel):
    """
    Plot some global network metric against another variable with associated
    y-axis errors for two network models.
    
    Parameters
    ----------
        x1: ndarray
            the x-axis points along which data was collected for network 1
        y1: ndarray
            the dependent global metric for network 1
        x_err1: ndarray
            the standard errors on the x-axis variable for network 1
        y_err1: ndarray
            the standard errors on the means of the global metrics for network 1
        x2: ndarray
            the x-axis points along which data was collected for network 2
        y2: ndarray
            the dependent global metric for network 2
        x_err2: ndarray
            the standard errors on the x-axis variable for network 2
        y_err2: ndarray
            the standard errors on the means of the global metrics for network 2
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
    fig, ax = plt.subplots(1, 1, layout='constrained')

    ax.errorbar(x1, y1, xerr=x_err1, yerr=y_err1, marker='o', linestyle='-', color=[172./255., 0, 64./255.], label='ER')
    ax.errorbar(x2, y2, xerr=x_err2, yerr=y_err2, marker='o', linestyle='-', color=[0, 80./255., 114./255.], label='BA')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.5)
    ax.legend()

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
        r'\# connected components, $n_c$',
        'degree Fano factor, $F$',
        'algebraic connectivity, $a$',
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
                ax.plot(df['p'], y_analytical, color=[205./255., 90./255., 19./255.], linestyle='--', label='analytical')
                ax.legend(loc='upper center')
            elif metric == 'avg_cc_mean':
                y_analytical = df['p']
                ax.plot(df['p'], y_analytical, color=[205./255., 90./255., 19./255.], linestyle='--', label='analytical')
                ax.legend(loc='upper center')
        elif model == 'BA' and metric == 'avg_k_mean':
            y_analytical = (2 * N - df['p'] - 1) * df['p'] / N
            ax.plot(df['p'], y_analytical, color=[205./255., 90./255., 19./255.], linestyle='--', label='analytical')
            ax.legend(loc='upper center')

        # add zero line to degree assortativity plot
        if metric == 'k_assortativity_mean':
            ax.axhline(0, color='black', linestyle='--', alpha=0.75)

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

def plot_ER_and_BA(N, n):
    """
    Plot comparisons of global metrics against average degree between the
    ER and BA models.
    
    Parameters
    ----------
        N: int
            network size
        n: int
            number of graphs sampled for the mean
    """
    # load in network data
    df_ER = pd.read_csv(f'graphs/data/ER/combined/{N}.txt')
    df_BA = pd.read_csv(f'graphs/data/BA/combined/{N}.txt')

    # set xlabel to average degree
    xlabel = r'average degree, $\langle k \rangle$'

    ylabels = [
        r'average clustering coefficient, $\langle C \rangle$',
        'transitivity, $T$',
        'average path length, $l$',
        'diameter, $d$',
        'degree assortativity, $r$',
        r'\# connected components, $n_c$',
        'degree Fano factor, $F$',
        'algebraic connectivity, $a$',
        r'spectral radius, $\rho$'
    ]

    # iterate over each metric in the dataframe
    for metric, error, ylabel in zip(df_ER.columns[3::2], df_ER.columns[4::2], ylabels):
        # plot figure
        fig, ax = plot_2_in_1_y_vs_x_with_errors(
            df_ER['avg_k_mean'], 
            df_ER[metric], 
            df_ER['avg_k_sem'], 
            df_ER[error], 
            df_BA['avg_k_mean'], 
            df_BA[metric], 
            df_BA['avg_k_sem'], 
            df_BA[error], 
            xlabel, 
            ylabel
        )

        # add zero line to degree assortativity plot
        if metric == 'k_assortativity_mean':
            ax.axhline(0, color='black', linestyle='--', alpha=0.75)

        # define filepath and save figure
        filename = f'./graphs/plots/ER+BA'
        os.makedirs(filename, exist_ok=True)
        plt.savefig(f'{filename}/{metric} {N}')

    # also plot standard deviation of degree assortativity
    k_ass_std_ER = df_ER['k_assortativity_sem'] * np.sqrt(n)
    k_ass_std_BA = df_BA['k_assortativity_sem'] * np.sqrt(n)
    ylabel = r'standard deviation of $r$, $\sigma_r$'
    fig, ax = plot_2_in_1_y_vs_x_with_errors(
        df_ER['avg_k_mean'], 
        k_ass_std_ER, 
        df_ER['avg_k_sem'],
        0, 
        df_BA['avg_k_mean'], 
        k_ass_std_BA, 
        df_BA['avg_k_sem'],
        0, 
        xlabel, 
        ylabel
    )
    # define filepath and save figure
    filename = f'./graphs/plots/ER+BA'
    os.makedirs(filename, exist_ok=True)
    plt.savefig(f'{filename}/{'k_assortativity_std'} {N}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=5000, help='network size')
    parser.add_argument('--all', action='store_true', help='plot all metrics for a 1D parameter space')
    parser.add_argument('--all2', action='store_true', help='plot all metrics for a both ER and BA models against <k>')
    parser.add_argument('-m', type=str, choices=['ER', 'BA'], help='Choose network model to generate')
    parser.add_argument('-n', type=int, default=10, help='number of graphs per parameter point')
    args = parser.parse_args()

    if args.all:
        plot_all_1d(args.m, args.N, args.n)
    elif args.all2:
        plot_ER_and_BA(args.N, args.n)