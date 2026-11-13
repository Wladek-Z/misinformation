import numpy as np
import networkx as nx
from epidemic_rds import RDS
import os
import argparse

# define transition probabilities for stages 1 and 2, assuming k = 50
k = 50
beta1 = np.zeros(9)
beta1[0:3] = [0.34 / k, 1 / 24, 0.0009 / k]
beta2 = np.array([0.02 / k, 0.02, 0.0009 / k, 0.0009 / k, 0.15 / k, 0.04, 0.2 / k, 0.4 / k, 0.0009 / k])

def run_sim(n, t, stage, model, N, p, q=None, K=None, instances=1, jnstances=1):
    """
    Run the RDS simulation for a given network, stage, and initial state.
    Save results to npz file.
    
    Parameters
    ----------
    n: ndarray
        number of nodes in each state at t=0
    t: int
        number of sweeps over which to run the simulation
    stage: int
        stage of the rumour life-cycle
    model: str
        network model to use
    N: int
        network size
    p: float
        first parameter for the network model
    q: float, optional
        second parameter for the network model
    K: int, optional
        number of groups for the DCSB model
    instances: int, default=1
        instances of the network to use
    jnstances: int, default=1
        instances of the simulation to run for each network

    Returns
    -------
    state_fracs: ndarray
        fraction of nodes in each state over time
    """
    # select parameters based on stage
    if stage == 1:
        beta = beta1
    elif stage == 2:
        beta = beta2

    # simulate over each instance
    for i in range(instances):
        # read in the network
        if model == 'ER' or model == 'BA' or model == 'REG':
            f = f'graphs/pajek/{model}/{i}/{N} {''.join(p.split('.'))}.net.gz'
            G = nx.read_pajek(f)
        elif model == 'SWSF':
            f = f'graphs/pajek/{model}/{i}/{N} {''.join(p.split('.'))} {''.join(q.split('.'))}.net.gz'
            G = nx.read_pajek(f)
        elif model == 'DCSB':
            f = f'graphs/pajek/{model}/{i}/{N} {K} {''.join(p.split('.'))} {''.join(q.split('.'))}.net.gz'
            G = nx.read_pajek(f)

        # repeat simulation for each network instance
        for j in range(jnstances):
            # print status
            print(f'Running network {i}, simulation {j}')
            # write output path
            if model == 'ER' or model == 'BA' or model == 'REG':
                output = f'epidemics/data/{model}/{i}/{j}/{N} {''.join(p.split('.'))} stage{stage} nS{n[1]} nD{n[2]}'
            elif model == 'SWSF':
                output = f'epidemics/data/{model}/{i}/{j}/{N} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{n[1]} nD{n[2]}'
            elif model == 'DCSB':
                output = f'epidemics/data/{model}/{i}/{j}/{N} {K} {''.join(p.split('.'))} {''.join(q.split('.'))} stage{stage} nS{n[1]} nD{n[2]}'
        
            # run the RDS simulation
            rds = RDS(G, beta, n, t)
            rds.run()

            # save the results to a npz file
            os.makedirs(f'epidemics/data/{model}/{i}/{j}', exist_ok=True)
            np.savez_compressed(output, state_fracs=rds.state_fracs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=5000, help='network size')
    parser.add_argument('-n', nargs='+', type=int, default=[4999, 1, 0, 0, 0], help='number of nodes in each state at t=0')
    parser.add_argument('-t', type=int, default=2000, help='number of sweeps')
    parser.add_argument('-p', type=str, help='first graph input parameter')
    parser.add_argument('-q', type=str, help='second graph input parameter')
    parser.add_argument('-K', type=int, default=10, help='number of groups for DCSBM')
    parser.add_argument('-m', type=str, choices=['ER', 'BA', 'DCSB', 'SWSF', 'REG'], help='network model to simulate')
    parser.add_argument('-s', type=int, choices=[1, 2], help='stage of the rumour life-cycle')
    parser.add_argument('-i', type=int, default=1, help='number of network instances to run')
    parser.add_argument('-j', type=int, default=1, help='number of simulation instances to run per network')
    parser.add_argument('--run', action='store_true', help='run the simulation')
    args = parser.parse_args()

    if args.run:
        run_sim(args.n, args.t, args.s, args.m, args.N, args.p, args.q, args.K, args.i, args.j)
