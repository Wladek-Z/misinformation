from matplotlib import pyplot as plt
import pandas as pd
import argparse
import numpy as np
import shendrukGroupFormat as ed
MYLW=1.0 #line width

from pathlib import Path

STYLE_FILE = Path(__file__).with_name("shendrukGroupStyle.mplstyle")
IMAGE_ROOT = Path.home() / "OneDrive - University of Edinburgh" / "MPhys" / "misinformation images" / "plots"
plt.style.use(str(STYLE_FILE))

def save_plot(filename):
    """Save a plot below the configured image directory."""
    output_path = IMAGE_ROOT / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)

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

    ax.errorbar(x, y, yerr=y_err, marker='o', linestyle='-', color=ed.saphire, label='numerical')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.grid(alpha=0.5)

    return fig, ax

def plot_heatmap(x, y, z, xlabel, ylabel, clabel):
    """
    Plot some global network metric in a 2-dimensional parameter space.
    
    Parameters
    ----------
        x: ndarray
            the first independent input parameter
        y: ndarray
            the second independent input parameter
        z: ndarray
            the metric data to plot
        xlabel: str
            the x-axis label
        ylabel: str
            the y-axis label    
        clabel: str
            the colour bar label
    
    Returns
    -------
        fig: matplotlib Figure
            the generated figure
        ax: matplotlib Axes
            the axis of the figure
    """
    # Sort by coordinates before reshaping so the input row order does not
    # determine which values are assigned to each heatmap cell.
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    if not (x.size == y.size == z.size):
        raise ValueError("x, y, and z must contain the same number of values")

    nx, ny = np.unique(x), np.unique(y)
    order = np.lexsort((y, x))
    size_x = len(nx)
    size_y = len(ny)
    Z = z[order].reshape(size_x, size_y).T
    X, Y = np.meshgrid(nx, ny)

    fig, ax = plt.subplots(1, 1, layout='constrained')

    heatmap = ax.pcolormesh(X, Y, Z, cmap=ed.inferno, shading='auto')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(heatmap, ax=ax, label=clabel)

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

    ax.errorbar(x1, y1, xerr=x_err1, yerr=y_err1, marker='o', linestyle='-', color=ed.saphire, label='ER')
    ax.errorbar(x2, y2, xerr=x_err2, yerr=y_err2, marker='o', linestyle='-', color=ed.crimson, label='BA')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.grid(alpha=0.5)
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
        'degree Fano factor, $F_k$',
        'clustering Fano factor, $F_C$',
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
                ax.plot(df['p'], y_analytical, color=ed.amber, linestyle='--', label='analytical')
                ax.legend(loc='upper center')
            elif metric == 'avg_cc_mean':
                y_analytical = df['p']
                ax.plot(df['p'], y_analytical, color=ed.amber, linestyle='--', label='analytical')
                ax.legend(loc='upper center')
        elif model == 'BA' and metric == 'avg_k_mean':
            y_analytical = (2 * N - df['p'] - 1) * df['p'] / N
            ax.plot(df['p'], y_analytical, color=ed.amber, linestyle='--', label='analytical')
            ax.legend(loc='upper center')

        # add zero line to degree assortativity plot
        if metric == 'k_assortativity_mean':
            ax.axhline(0, color='black', linestyle='--', alpha=0.75)

        # define filepath and save figure
        save_plot(Path(model) / f'{metric} {N}')

    # also plot standard deviation of degree assortativity
    k_ass_std = df['k_assortativity_sem'] * np.sqrt(n)
    ylabel = r'standard deviation of $r$, $\sigma_r$'
    fig, ax = plot_y_vs_x_with_errors(df['p'], k_ass_std, 0, xlabel, ylabel)
    # define filepath and save figure
    save_plot(Path(model) / f'k_assortativity_std {N}')

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
        'degree Fano factor, $F_k$',
        'clustering Fano factor, $F_C$',
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
        save_plot(Path('ER+BA') / f'{metric} {N}')

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
    save_plot(Path('ER+BA') / f'k_assortativity_std {N}')

def plot_ER_and_BA_log(N, n):
    """
    Plot comparisons of global metrics against average degree between the
    ER and BA models in log space.
    
    Parameters
    ----------
        N: int
            network size
        n: int
            number of graphs sampled for the mean
    """
    # load in network data
    df_ER = pd.read_csv(f'graphs/data/ER/combined/{N} extra.txt')
    df_BA = pd.read_csv(f'graphs/data/BA/combined/{N} extra.txt')

    # set xlabel to average degree
    xlabel = r'average degree, $\langle k \rangle$'

    ylabels = [
        r'average clustering coefficient, $\langle C \rangle$',
        'transitivity, $T$',
        'average path length, $l$',
        'diameter, $d$',
        'degree assortativity, $r$',
        r'\# connected components, $n_c$',
        'degree Fano factor, $F_k$',
        'clustering Fano factor, $F_C$',
        'algebraic connectivity, $a$',
        r'spectral radius, $\rho$'
    ]

    # iterate over each metric in the dataframe
    for metric, error, ylabel in zip(df_ER.columns[3::2], df_ER.columns[4::2], ylabels):
        # skip number of connected components and degree assortativity (useless in log space)
        if metric == 'n_components_mean' or metric == 'k_assortativity_mean':
            continue
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
        # convert to logscale
        ax.set_xscale('log')
        ax.set_yscale('log')

        # define filepath and save figure
        save_plot(Path('ER+BA') / 'log' / f'{metric} {N}')

    # also plot difference between clustering coefficient and transitivity
    err_ER = np.sqrt(df_ER['avg_cc_sem']**2 + df_ER['transitivity_sem']**2)
    err_BA = np.sqrt(df_BA['avg_cc_sem']**2 + df_BA['transitivity_sem']**2)
    ylabel = r'$|\langle C \rangle - T|$'
    fig, ax = plot_2_in_1_y_vs_x_with_errors(
        df_ER['avg_k_mean'], 
        np.abs(df_ER['avg_cc_mean'] - df_ER['transitivity_mean']), 
        df_ER['avg_k_sem'],
        err_ER, 
        df_BA['avg_k_mean'], 
        np.abs(df_BA['avg_cc_mean'] - df_BA['transitivity_mean']), 
        df_BA['avg_k_sem'],
        err_BA,
        xlabel, 
        ylabel
    )
    # convert to logscale
    ax.set_xscale('log')
    ax.set_yscale('log')
    # define filepath and save figure
    save_plot(Path('ER+BA') / 'log' / f'cc_T_diff {N}')

def plot_all_2d(model, N, n, K=None):
    """
    Plot all global metrics in a 2-dimensional parameter space for
    a given double parameter network model of size N.
    
    Parameters
    ----------
        model: str
            which network model to plot results for
        N: int
            size of network model used in results
        n: int
            number of graphs used to calculate mean
        K: int
            number of groups, if applicable
    """
    # load in network data
    if model == 'DCSB':
        df = pd.read_csv(f'graphs/data/{model}/combined/{N} {K}.txt')
    elif model == 'SWSF':
        df = pd.read_csv(f'graphs/data/{model}/combined/{N}.txt')

    # set axis labels
    xlabel = 'edges per new node, $m$'
    if model == 'DCSB':
        ylabel = r'mixing parameter, $\nu$'
    elif model == 'SWSF':
        ylabel = r'deactivated edge probability, $\mu$'

    # set colour bar labels
    clabels = {
        'avg_k_mean': r'average degree, $\langle k \rangle$',
        'avg_cc_mean': r'average clustering coefficient, $\langle C \rangle$',
        'transitivity_mean': 'transitivity, $T$',
        'avg_path_length_mean': 'average path length, $l$',
        'diameter_mean': 'diameter, $d$',
        'k_assortativity_mean': 'degree assortativity, $r$',
        'n_components_mean': r'\# connected components, $n_c$',
        'F_factor_k_mean': 'degree Fano factor, $F_k$',
        'F_factor_cc_mean': 'clustering Fano factor, $F_C$',
        'a_connectivity_mean': 'algebraic connectivity, $a$',
        'spectral_r_mean': r'spectral radius, $\rho$',
        'modularity_mean': 'modularity, $Q$',
    }

    # Select metric means explicitly
    metrics = [column for column in df.columns if column in clabels]
    # iterate over each metric
    for metric in metrics:
        clabel = clabels[metric]
        # plot figure
        fig, ax = plot_heatmap(df['p'], df['q'], df[metric], xlabel, ylabel, clabel)

        # define filepath and save figure
        if model == 'DCSB':
            save_plot(Path(model) / f'{metric} {N} {K}')
        elif model == 'SWSF':
            save_plot(Path(model) / f'{metric} {N}')

    # also plot standard deviation of degree assortativity
    k_ass_std = df['k_assortativity_sem'] * np.sqrt(n)
    clabel = r'standard deviation of $r$, $\sigma_r$'
    fig, ax = plot_heatmap(df['p'], df['q'], k_ass_std, xlabel, ylabel, clabel)
    # define filepath and save figure
    if model == 'DCSB':
        save_plot(Path(model) / f'k_ass_std {N} {K}')
    elif model == 'SWSF':
        save_plot(Path(model) / f'k_ass_std {N}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=5000, help='network size')
    parser.add_argument('-K', type=int, default=10, help='number of groups')
    parser.add_argument('--all', action='store_true', help='plot all metrics for a 1D parameter space')
    parser.add_argument('--all2', action='store_true', help='plot all metrics for a both ER and BA models against <k>')
    parser.add_argument('--log2', action='store_true', help='plot all metrics for a both ER and BA models in logspace')
    parser.add_argument('--all2d', action='store_true', help='plot all metrics on 2-dimensional heat maps')
    parser.add_argument('-m', type=str, choices=['ER', 'BA', 'SWSF', 'DCSB'], help='Choose network model to generate')
    parser.add_argument('-n', type=int, default=10, help='number of graphs per parameter point')
    args = parser.parse_args()

    if args.all:
        plot_all_1d(args.m, args.N, args.n)
    elif args.all2:
        plot_ER_and_BA(args.N, args.n)
    elif args.all2d:
        plot_all_2d(args.m, args.N, args.n, args.K)
    elif args.log2:
        plot_ER_and_BA_log(args.N, args.n)