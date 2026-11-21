from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
from io import BytesIO
import shendrukGroupFormat as ed
MYLW=1.0 #line width

from pathlib import Path

STYLE_FILE = Path(__file__).with_name("shendrukGroupStyle.mplstyle")
IMAGE_ROOT = Path.home() / "OneDrive - University of Edinburgh" / "MPhys" / "misinformation images" / "epidemic plots"
plt.style.use(str(STYLE_FILE))

def read_data(model, N, stage, nS, nD, p, q=None, K=None, instance=0, jnstance=0, initcon=None):
    """
    Read in the simulation results for a given network and simulation instance.
    
    Parameters
    ----------
    model: str
        network model used in the simulation
    N: int
        network size
    stage: int
        stage of the rumour life-cycle
    nS: int
        number of initial spreaders
    nD: int
        number of initial debunkers
    p: float
        first parameter for the network model
    q: float, optional
        second parameter for the network model
    K: int, optional
        number of groups for the DCSB model
    instance: int, default=0
        instance of the network used in the simulation
    jnstance: int, default=0
        instance of the simulation used in the simulation
    initcon: str, optional
        initial condition for running specific types of simulation
    
    Returns
    -------
    ndarray
        fraction of nodes in each state over time
    """
    # resolve filepath to the simulation results
    if model == 'ER' or model == 'BA' or model == 'REG':
        if initcon is not None:
            f = f'epidemics/data/{model}/{initcon}/{instance}/{jnstance}/{N} {''.join(p.split('.'))} stage{stage} nS{nS} nD{nD}.npz'
        else:
            f = f'epidemics/data/{model}/{instance}/{jnstance}/{N} {''.join(p.split('.'))} stage{stage} nS{nS} nD{nD}.npz'
    elif model == 'SWSF':
        if initcon is not None:
            f = f'epidemics/data/{model}/{initcon}/{instance}/{jnstance}/{N} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}.npz'
        else:
            f = f'epidemics/data/{model}/{instance}/{jnstance}/{N} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}.npz'
    elif model == 'DCSB':
        if initcon is not None:
            f = f'epidemics/data/{model}/{initcon}/{instance}/{jnstance}/{N} {K} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}.npz'
        else:
            f = f'epidemics/data/{model}/{instance}/{jnstance}/{N} {K} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}.npz'

    # load the simulation results
    data = np.load(f)
    return data['state_fracs']

def save_plot(filename):
    """
    Save a plot below the configured image directory.

    Parameters
    ----------
    filename: str
        the filename to save the plot as, relative to the image root directory
    """
    output_path = IMAGE_ROOT / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='png')

def inverse_crossover_time(state_fracs):
    """
    Return the inverse time at which S_4 overtakes S_1.

    Parameters
    ----------
    state_fracs: ndarray
        fraction of nodes in each state over time
    
    Returns
    -------
    float
        inverse time at which S_4 overtakes S_1, or 0 if it never does
    """
    crossing_times = np.flatnonzero(state_fracs[:, 3] > state_fracs[:, 0])
    if crossing_times.size == 0:
        return 0.0

    crossover_time = crossing_times[0]
    return 1.0 / crossover_time if crossover_time != 0 else 0.0

def calculate_all_metrics(state_fracs):
    """
    Calculate all metrics of interest from the state fractions.

    Parameters
    ----------
    state_fracs: ndarray
        fraction of nodes in each state over time
    
    Returns
    -------
    dict
        dictionary containing all metrics of interest
    """
    metrics = {
        "R": state_fracs[-1, 3],
        "max_S": np.max(state_fracs[:, 1]),
        "t_ext": len(state_fracs),
        "t_peak": np.argmax(state_fracs[:, 1]),
        "inv_tc": inverse_crossover_time(state_fracs),
        "max_dS4": np.max(np.abs(np.diff(state_fracs[:, 3]))),
        "max_dS1": np.max(np.abs(np.diff(state_fracs[:, 0]))),
    }
    return metrics

def plot_state_fractions(model, N, stage, nS, nD, p, q=None, K=None, instance=0, jnstance=0):
    """
    Plot the fraction of nodes in each state over time for a given simulation.
    
    Parameters
    ----------
    model: str
        network model used in the simulation
    N: int
        network size
    stage: int
        stage of the rumour life-cycle
    nS: int
        number of initial spreaders
    nD: int
        number of initial debunkers
    p: float
        first parameter for the network model
    q: float, optional
        second parameter for the network model
    K: int, optional
        number of groups for the DCSB model
    instance: int, default=0
        instance of the network used in the simulation
    jnstance: int, default=0
        instance of the simulation used in the simulation
    """
    # resolve filepath to the simulation results
    if model == 'ER' or model == 'BA' or model == 'REG':
        output = f'{model}/fracs/{instance}/{jnstance}/{N} {''.join(p.split('.'))} stage{stage} nS{nS} nD{nD}.png'
    elif model == 'SWSF':
        output = f'{model}/fracs/{instance}/{jnstance}/{N} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}.png'
    elif model == 'DCSB':
        output = f'{model}/fracs/{instance}/{jnstance}/{N} {K} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}.png'

    # load the simulation results
    state_fracs = read_data(model, N, stage, nS, nD, p, q, K, instance, jnstance)

    # get time
    t = np.arange(state_fracs.shape[0])

    # plot the state fractions over time
    fig = plt.figure(figsize=(8, 6), layout='constrained')
    plt.xlabel('time (sweeps)')
    plt.ylabel('fraction')

    # initialise colours
    colours = [ed.saphire, ed.crimson, ed.capri, ed.amber, ed.plum]

    # plot the state fractions over time
    for i, state_frac in enumerate(state_fracs.T):
        if np.sum(state_frac) > 0:
            plt.plot(t, state_frac, label=f'$S_{i+1}$', color=colours[i])

    plt.legend()

    # save the plot
    save_plot(output)

    # final fraction of rumour stiflers
    print(f"Final fraction of rumour stiflers: R = {state_fracs[-1, 3]}")

    # maximal scale of spreaders
    print(f"Maximal scale of spreaders: max(S_2) = {np.max(state_fracs[:, 1])}")

    # time to extinction
    print(f"Time to extinction: t_ext = {len(state_fracs)} sweeps")

    # time to spreader peak
    print(f"Time to Spreader peak: t_peak = {np.argmax(state_fracs[:, 1])} sweeps")

    # inverse time for S_4 to overtake S_1
    inv_tc = inverse_crossover_time(state_fracs)
    print(f"Inverse crossover time (S_4 > S_1): 1/t_c = {inv_tc} sweeps^-1")

    # maximum slope of S_4
    print(f"Maximum slope of S_4: max(|dS_4/dt|) = {np.max(np.abs(np.diff(state_fracs[:, 3])))} sweeps^-1")

    # maximum slope of S_2
    print(f"Maximum slope of S_2: max(|dS_2/dt|) = {np.max(np.abs(np.diff(state_fracs[:, 1])))} sweeps^-1")

def plot_variability(model, N, stage, nS, nD, p, q=None, K=None, instances=1, jnstances=1):
    """
    Compare the variability in results across instances of the network and instances of the simulation.
    
    Parameters
    ----------
    model: str
        network model used in the simulation
    N: int
        network size
    stage: int
        stage of the rumour life-cycle
    nS: int
        number of initial spreaders
    nD: int
        number of initial debunkers
    p: str
        first graph input parameter
    q: str, optional
        second graph input parameter
    K: int, optional
        number of groups for DCSBM
    instances: int, optional
        number of instances of the network used
    jnstances: int, optional
        number of instances of the simulation used
    """
    # initialise arrays to store results and errors
    R_list = np.empty((instances, jnstances))
    max_S_list = np.empty((instances, jnstances))
    t_ext_list = np.empty((instances, jnstances))
    t_peak_list = np.empty((instances, jnstances))
    R_sems = np.empty(instances)
    max_S_sems = np.empty(instances)
    t_ext_sems = np.empty(instances)
    t_peak_sems = np.empty(instances)
    

    # read in results for each instance of the network
    for i in range(instances):
        # read in results for each instance of the simulation
        for j in range(jnstances):
            # load in simulation results
            state_fracs = read_data(model, N, stage, nS, nD, p, q, K, i, j)

            # store final scope of rumour, maximal scale of spreaders, and time to extinction
            R_list[i, j] = state_fracs[-1, 3]
            max_S_list[i, j] = np.max(state_fracs[:, 1])
            t_ext_list[i, j] = len(state_fracs)
            t_peak_list[i, j] = np.argmax(state_fracs[:, 1])

        # compute standard error on the mean for each instance of the network
        R_sems[i] = np.std(R_list[i, :], ddof=1) / np.sqrt(jnstances)
        max_S_sems[i] = np.std(max_S_list[i, :], ddof=1) / np.sqrt(jnstances)
        t_ext_sems[i] = np.std(t_ext_list[i, :], ddof=1) / np.sqrt(jnstances)
        t_peak_sems[i] = np.std(t_peak_list[i, :], ddof=1) / np.sqrt(jnstances)

    def plot(y, yerr, ylabel, filename):
        """
        Plot the simulation instance results, the mean result, and the error on the mean for each network instance.
        """
        # initialise the figure
        fig = plt.figure(figsize=(8, 6), layout='constrained')
        plt.xlabel('network instance')
        plt.ylabel(ylabel)
        plt.xticks(np.arange(instances))
        # create list of colours from cmap
        colours = plt.cm.tab10.colors

        # iterate over each network instance
        for i in range(instances):
            # plot results of each simulation instance for each network instance
            plt.scatter([i] * jnstances, y[i, :], color=colours[i], alpha=0.5, zorder=1)
            # plot means and errors for each network instance
            plt.errorbar(i, np.mean(y[i, :]), yerr=yerr[i], fmt='o', markersize=12, color='black', zorder=2)

        # calculate mean y across network instances
        y_networks = np.mean(y, axis=1)
        network_mean = np.mean(y_networks)
        network_sem = np.std(y_networks, ddof=1) / np.sqrt(instances)

        # draw overall mean and error across network instances
        plt.axhline(network_mean, color='r', linestyle='--', linewidth=2, zorder=0)
        plt.axhspan(network_mean - network_sem, network_mean + network_sem, color='r', alpha=0.2)

        # save the plot
        save_plot(filename)

    # plot final fraction of rumour stiflers
    if model == 'ER' or model == 'BA' or model == 'REG':
        f = f'{model}/variability/{N} {''.join(p.split('.'))} stage{stage} nS{nS} nD{nD}'
    elif model == 'SWSF':
        f = f'{model}/variability/{N} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}'
    elif model == 'DCSB':
        f = f'{model}/variability/{N} {K} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}'

    # plot and save the results
    plot(R_list, R_sems, 'final scope of rumour, $R$', f'{f}/R.png')
    plot(max_S_list, max_S_sems, r'max$(S_2)$', f'{f}/max_S.png')
    plot(t_ext_list, t_ext_sems, r'time to extinction, $t_{\mathrm{ext}}$', f'{f}/t_ext.png')
    plot(t_peak_list, t_peak_sems, r'time to Spreader peak, $t_{\mathrm{peak}}$', f'{f}/t_peak.png')

def plot_variability_pdf(model, N, stage, nS, nD, p, q=None, K=None, instances=1, jnstances=1):
    """
    Compare the variability in results across instances of the simulation as probability distributions.
    
    Parameters
    ----------
    model: str
        network model used in the simulation
    N: int
        network size
    stage: int
        stage of the rumour life-cycle
    nS: int
        number of initial spreaders
    nD: int
        number of initial debunkers
    p: str
        first graph input parameter
    q: str, optional
        second graph input parameter
    K: int, optional
        number of groups for DCSBM
    instances: int, default=1
        number of instances of the network used
    jnstances: int, default=1
        number of instances of the simulation used
    """
    # initialise arrays to store results
    R_list = np.empty((instances, jnstances))
    max_S_list = np.empty((instances, jnstances))
    t_ext_list = np.empty((instances, jnstances))
    t_peak_list = np.empty((instances, jnstances))
    inv_tc_list = np.empty((instances, jnstances))
    max_dS4_list = np.empty((instances, jnstances))
    max_dS1_list = np.empty((instances, jnstances))

    # number of bins for histograms
    NBINS = 20

    # read in results for each instance of the network
    for i in range(instances):
        # read in results for each instance of the simulation
        for j in range(jnstances):
            # load in simulation results
            state_fracs = read_data(model, N, stage, nS, nD, p, q, K, i, j)
            # calculate metrics
            metrics = calculate_all_metrics(state_fracs)
            # store metrics of interest
            R_list[i, j] = metrics["R"]
            max_S_list[i, j] = metrics["max_S"]
            t_ext_list[i, j] = metrics["t_ext"]
            t_peak_list[i, j] = metrics["t_peak"]
            inv_tc_list[i, j] = metrics["inv_tc"]
            max_dS4_list[i, j] = metrics["max_dS4"]
            max_dS1_list[i, j] = metrics["max_dS1"]

    def plot(Y, ylabel, filename):
        """
        Plot the simulation instance results histogram and the mean histogram for each network instance.
        """
        Y = np.asarray(Y, dtype=float)
        bins = np.histogram_bin_edges(Y, bins=NBINS)

        # initialise the figure
        fig = plt.figure(figsize=(8, 6), layout='constrained')
        plt.xlabel(r'\# occurrences')
        plt.ylabel(ylabel)
        # create list of colours from cmap
        colours = plt.cm.tab10.colors
        histogram_counts = []

        # iterate over each network instance
        for i in range(instances):
            # plot overlapping, horizontally-aligned histograms of results for each network instance
            counts, bin_edges = np.histogram(Y[i, :], bins=bins)
            histogram_counts.append(counts)
            plt.hist(Y[i, :], bins=bin_edges, alpha=0.5, color=colours[i], orientation='horizontal')

        # overlay the mean histogram across network instances
        mean_counts = np.mean(histogram_counts, axis=0)
        plt.stairs(mean_counts, bin_edges, orientation='horizontal', color='red', linestyle='--', linewidth=2)

        # save the plot
        save_plot(filename)

    # resolve filepath to figure output
    if model == 'ER' or model == 'BA' or model == 'REG':
        f = f'{model}/variability/PDF/{N} {''.join(p.split('.'))} stage{stage} nS{nS} nD{nD}'
    elif model == 'SWSF':
        f = f'{model}/variability/PDF/{N} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}'
    elif model == 'DCSB':
        f = f'{model}/variability/PDF/{N} {K} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}'

    # plot and save the results
    plot(R_list, 'final scope of rumour, $R$', f'{f}/R.png')
    plot(max_S_list, r'max$(S_2)$', f'{f}/max_S.png')
    plot(t_ext_list, r'time to extinction, $t_{\mathrm{ext}}$', f'{f}/t_ext.png')
    plot(t_peak_list, r'time to Spreader peak, $t_{\mathrm{peak}}$', f'{f}/t_peak.png')
    plot(inv_tc_list, r'inverse crossover time, $1/t_c$', f'{f}/inv_tc.png')
    plot(max_dS4_list, r'max$(|dS_4/dt|)$', f'{f}/max_dS4.png')
    plot(max_dS1_list, r'max$(|dS_1/dt|)$', f'{f}/max_dS1.png')

def plot_randomspreader_variability(model, N, stage, nS, nD, p, q=None, K=None, instances=1, jnstances=1, i_rs=0, j_rs=1):
    """
    Compare the variability in results across instances of the simulation for fixed random spreader in a specific network.
    
    Parameters
    ----------
    model: str
        network model used in the simulation
    N: int
        network size
    stage: int
        stage of the rumour life-cycle
    nS: int
        number of initial spreaders
    nD: int
        number of initial debunkers
    p: str
        first graph input parameter
    q: str, optional
        second graph input parameter
    K: int, optional
        number of groups for DCSBM
    instances: int, default=1
        number of instances of the network used
    jnstances: int, default=1
        number of instances of the simulation used
    i_rs: int, default=0
        instance of the network used in randomspreader
    j_rs: int, default=1
        instances of the simulation used in randomspreader
    """
    if j_rs < 1:
        raise ValueError("j_rs must be at least 1")
    if not 0 <= i_rs < instances:
        raise ValueError("i_rs must identify one of the network instances")

    # initialise arrays to store results for fixed random spreader simulations
    R_list_rs = np.empty(j_rs)
    max_S_list_rs = np.empty(j_rs)
    t_ext_list_rs = np.empty(j_rs)
    t_peak_list_rs = np.empty(j_rs)
    inv_tc_list_rs = np.empty(j_rs)
    max_dS4_list_rs = np.empty(j_rs)
    max_dS1_list_rs = np.empty(j_rs)

    # initialise arrays to store results for histograms for unfixed random spreader simulations
    R_list = np.empty((instances, jnstances))
    max_S_list = np.empty((instances, jnstances))
    t_ext_list = np.empty((instances, jnstances))
    t_peak_list = np.empty((instances, jnstances))
    inv_tc_list = np.empty((instances, jnstances))
    max_dS4_list = np.empty((instances, jnstances))
    max_dS1_list = np.empty((instances, jnstances))

    # number of bins for histograms
    NBINS = 20

    # read in fixed-random-spreader results
    for j in range(j_rs):
        state_fracs = read_data(model, N, stage, nS, nD, p, q, K, i_rs, j, initcon='randomspreader')
        metrics = calculate_all_metrics(state_fracs)
        R_list_rs[j] = metrics["R"]
        max_S_list_rs[j] = metrics["max_S"]
        t_ext_list_rs[j] = metrics["t_ext"]
        t_peak_list_rs[j] = metrics["t_peak"]
        inv_tc_list_rs[j] = metrics["inv_tc"]
        max_dS4_list_rs[j] = metrics["max_dS4"]
        max_dS1_list_rs[j] = metrics["max_dS1"]

    # read in results for each instance of the network
    for i in range(instances):
        # read in results for each instance of the simulation
        for j in range(jnstances):
            # load in simulation results
            state_fracs = read_data(model, N, stage, nS, nD, p, q, K, i, j)
            # calculate metrics
            metrics = calculate_all_metrics(state_fracs)
            # store metrics of interest
            R_list[i, j] = metrics["R"]
            max_S_list[i, j] = metrics["max_S"]
            t_ext_list[i, j] = metrics["t_ext"]
            t_peak_list[i, j] = metrics["t_peak"]
            inv_tc_list[i, j] = metrics["inv_tc"]
            max_dS4_list[i, j] = metrics["max_dS4"]
            max_dS1_list[i, j] = metrics["max_dS1"]

    def plot(Y, y, ylabel, filename):
        """
        Plot the simulation instance results histogram and the mean histogram for each network instance.
        Overlay scatterplot results for fixed random spreader simulations.
        """
        Y = np.asarray(Y, dtype=float)
        bins = np.histogram_bin_edges(Y, bins=NBINS)
        y = np.asarray(y, dtype=float)

        # initialise the figure
        fig = plt.figure(figsize=(8, 6), layout='constrained')
        plt.xlabel(r'\# occurrences')
        plt.ylabel(ylabel)
        # create list of colours from cmap
        colours = plt.cm.tab10.colors
        histogram_counts = []

        # iterate over each network instance
        for i in range(instances):
            # plot overlapping, horizontally-aligned histograms of results for each network instance
            counts, bin_edges = np.histogram(Y[i, :], bins=bins)
            histogram_counts.append(counts)
            plt.hist(Y[i, :], bins=bin_edges, alpha=0.5, color=colours[i], orientation='horizontal')

        # overlay the mean histogram across network instances
        mean_counts = np.mean(histogram_counts, axis=0)
        plt.stairs(mean_counts, bin_edges, orientation='horizontal', color='red', linestyle='--', linewidth=2, label='all networks')
        plt.axhline(np.mean(Y), color='red', linestyle='dotted', linewidth=2, label='mean of all networks')

        # scatter plot results for fixed random spreader simulations
        plt.scatter(-np.ones(len(y)), y, color='black', marker='x', s=100, label='fixed spreader')
        plt.axhline(np.mean(y), color='black', linestyle='dotted', linewidth=2, label='mean of fixed spreader')

        # add legend
        plt.legend()
        # save the plot
        save_plot(filename)

    # resolve filepath to figure output
    if model == 'ER' or model == 'BA' or model == 'REG':
        f = f'{model}/variability/randomspreader/{N} {''.join(p.split('.'))} stage{stage} nS{nS} nD{nD}'
    elif model == 'SWSF':
        f = f'{model}/variability/randomspreader/{N} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}'
    elif model == 'DCSB':
        f = f'{model}/variability/randomspreader/{N} {K} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}'

    # plot and save the results
    plot(R_list, R_list_rs, 'final scope of rumour, $R$', f'{f}/R.png')
    plot(max_S_list, max_S_list_rs, r'max$(S_2)$', f'{f}/max_S.png')
    plot(t_ext_list, t_ext_list_rs, r'time to extinction, $t_{\mathrm{ext}}$', f'{f}/t_ext.png')
    plot(t_peak_list, t_peak_list_rs, r'time to Spreader peak, $t_{\mathrm{peak}}$', f'{f}/t_peak.png')
    plot(inv_tc_list, inv_tc_list_rs, r'inverse crossover time, $1/t_c$', f'{f}/inv_tc.png')
    plot(max_dS4_list, max_dS4_list_rs, r'max$(|dS_4/dt|)$', f'{f}/max_dS4.png')
    plot(max_dS1_list, max_dS1_list_rs, r'max$(|dS_1/dt|)$', f'{f}/max_dS1.png')    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=5000, help='network size')
    parser.add_argument('-p', type=str, help='first graph input parameter')
    parser.add_argument('-q', type=str, help='second graph input parameter')
    parser.add_argument('-K', type=int, default=10, help='number of groups for DCSBM')
    parser.add_argument('-m', type=str, choices=['ER', 'BA', 'DCSB', 'SWSF', 'REG'], help='network model to simulate')
    parser.add_argument('-s', type=int, choices=[1, 2], help='stage of the rumour life-cycle')
    parser.add_argument('-nS', type=int, default=1, help='number of initial spreaders')
    parser.add_argument('-nD', type=int, default=0, help='number of initial debunkers')
    parser.add_argument('-i', type=int, default=0, help='instance(s) of the network to use')
    parser.add_argument('-j', type=int, default=0, help='instance(s) of the simulation to use')
    parser.add_argument('-irs', type=int, default=0, help='instance of the network used in randomspreader')
    parser.add_argument('-jrs', type=int, default=1, help='instances of the simulation used in randomspreader')
    parser.add_argument('--fracs', action='store_true', help='plot state fractions over time')
    parser.add_argument('--variability', action='store_true', help='plot variability of results for a given set of network instances')
    parser.add_argument('--varPDF', action='store_true', help='plot variability PDFs of results for a given set of network instances')
    parser.add_argument('--randomspreader', action='store_true', help='plot variability of results for fixed random spreader')
    args = parser.parse_args()

    if args.fracs:
        plot_state_fractions(args.m, args.N, args.s, args.nS, args.nD, args.p, args.q, args.K, args.i)
    elif args.variability:
        plot_variability(args.m, args.N, args.s, args.nS, args.nD, args.p, args.q, args.K, args.i, args.j)
    elif args.varPDF:
        plot_variability_pdf(args.m, args.N, args.s, args.nS, args.nD, args.p, args.q, args.K, args.i, args.j)
    elif args.randomspreader:
        plot_randomspreader_variability(args.m, args.N, args.s, args.nS, args.nD, args.p, args.q, args.K, args.i, args.j, args.irs, args.jrs)