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

def read_data(model, N, stage, nS, nD, p, q=None, K=None, instance=0, jnstance=0):
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
    
    Returns
    -------
    ndarray
        fraction of nodes in each state over time
    """
    # resolve filepath to the simulation results
    if model == 'ER' or model == 'BA' or model == 'REG':
        f = f'epidemics/data/{model}/{instance}/{jnstance}/{N} {''.join(p.split('.'))} stage{stage} nS{nS} nD{nD}.npz'
    elif model == 'SWSF':
        f = f'epidemics/data/{model}/{instance}/{jnstance}/{N} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}.npz'
    elif model == 'DCSB':
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
    inv_tc = 1 / np.argmin(state_fracs[:, 3] > state_fracs[:, 0]) if np.any(state_fracs[:, 3] > state_fracs[:, 0]) else 0
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
    NBINS = 25 

    # read in results for each instance of the network
    for i in range(instances):
        # read in results for each instance of the simulation
        for j in range(jnstances):
            # load in simulation results
            state_fracs = read_data(model, N, stage, nS, nD, p, q, K, i, j)

            # store metrics of interest
            R_list[i, j] = state_fracs[-1, 3]
            max_S_list[i, j] = np.max(state_fracs[:, 1])
            t_ext_list[i, j] = len(state_fracs)
            t_peak_list[i, j] = np.argmax(state_fracs[:, 1])
            inv_tc_list[i, j] = 1 / np.argmin(state_fracs[:, 3] > state_fracs[:, 0]) if np.any(state_fracs[:, 3] > state_fracs[:, 0]) else 0
            max_dS4_list[i, j] = np.max(np.abs(np.diff(state_fracs[:, 3])))
            max_dS1_list[i, j] = np.max(np.abs(np.diff(state_fracs[:, 0])))

    def plot(Y, ylabel, filename):
        """
        Plot the simulation instance results histogram and the mean histogram for each network instance.
        """
        # initialise the figure
        fig = plt.figure(figsize=(8, 6), layout='constrained')
        plt.xlabel('probability density')
        plt.ylabel(ylabel)
        # create list of colours from cmap
        colours = plt.cm.tab10.colors

        # iterate over each network instance
        for i in range(instances):
            # plot overlapping, horizontally-aligned histograms of results for each network instance
            plt.hist(Y[i, :], bins=NBINS, density=True, alpha=0.4, color=colours[i], orientation='horizontal')

        # overlay the mean histogram across network instances
        plt.hist(np.mean(Y, axis=0), bins=NBINS, density=True, color='red', orientation='horizontal', histtype='step', linestyle='--')

        # save the plot
        save_plot(filename)

    # plot final fraction of rumour stiflers
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
    parser.add_argument('--fracs', action='store_true', help='plot state fractions over time')
    parser.add_argument('--variability', action='store_true', help='plot variability of results for a given network')
    args = parser.parse_args()

    if args.fracs:
        plot_state_fractions(args.m, args.N, args.s, args.nS, args.nD, args.p, args.q, args.K, args.i)
    elif args.variability:
        plot_variability(args.m, args.N, args.s, args.nS, args.nD, args.p, args.q, args.K, args.i, args.j)