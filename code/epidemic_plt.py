from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
import shendrukGroupFormat as ed
MYLW=1.0 #line width

from pathlib import Path

STYLE_FILE = Path(__file__).with_name("shendrukGroupStyle.mplstyle")
IMAGE_ROOT = Path.home() / "OneDrive - University of Edinburgh" / "MPhys" / "misinformation images" / "epidemic plots"
plt.style.use(str(STYLE_FILE))

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
    plt.savefig(output_path)

def plot_state_fractions(model, N, stage, nS, nD, p, q=None, K=None, instance=0):
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
    """
    # resolve filepath to the simulation results
    if model == 'ER' or model == 'BA':
        f = f'epidemics/data/{model}/{instance}/{N} {''.join(p.split('.'))} stage{stage} nS{nS} nD{nD}.npz'
        output = f'{model}/fracs/{instance}/{N} {''.join(p.split('.'))} stage{stage} nS{nS} nD{nD}.png'
    elif model == 'SWSF':
        f = f'epidemics/data/{model}/{instance}/{N} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}.npz'
        output = f'{model}/fracs/{instance}/{N} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}.png'
    elif model == 'DCSB':
        f = f'epidemics/data/{model}/{instance}/{N} {K} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}.npz'
        output = f'{model}/fracs/{instance}/{N} {K} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{nS} nD{nD}.png'

    # load the simulation results
    data = np.load(f)
    state_fracs = data['state_fracs']

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

    # print final fraction of rumour stiflers, maximal scale of spreaders, and time to extinction
    print(f"Final fraction of rumour stiflers: R = {state_fracs[-1, 3]}")
    print(f"Maximal scale of spreaders: max(S_2) = {np.max(state_fracs[:, 1])}")
    print(f"Time to extinction: t_ext = {len(state_fracs)} sweeps")

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
    parser.add_argument('-i', type=int, default=0, help='instance of the network to use')
    parser.add_argument('--fracs', action='store_true', help='plot state fractions over time')
    args = parser.parse_args()

    if args.fracs:
        plot_state_fractions(args.m, args.N, args.s, args.nS, args.nD, args.p, args.q, args.K, args.i)