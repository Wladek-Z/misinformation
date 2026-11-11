import numpy as np
import networkx as nx

class RDS():
    """
    class for simulating the RDS (rumour spreader-debunker) model on a networkx graph.
    """

    def __init__(self, G, beta, n, t):
        """
        Initialise the RDS model with the given parameters.
        
        Parameters
        ----------
        G: networkx Graph
            the graph on which to simulate the RDS model.
        beta: ndarray
            array of the 9 possible transition probabilities
        n: ndarray
            array of the number of nodes in each state at t=0
        t: int
            the number of sweeps over which to run the simulation
        """
        # initialise variables
        self.G = G
        self.beta = beta
        self.t = t

        # get size of graph
        self.N = G.number_of_nodes()

        # check for valid input
        if np.sum(n) != self.N:
            raise ValueError("The sum of the initial state counts must equal the number of nodes in the graph.")

        # track the fraction of nodes in each state over time
        self.state_fracs = np.zeros((t, 5))
        self.state_fracs[0] = np.array(n) / self.N

        # define possible states
        self.state_labels = ['S1', 'S2', 'S3', 'S4', 'S5']

        # populate the initial states of the nodes in the graph
        self.node_states = {}
        unassigned_nodes = list(self.G.nodes())

        for num, label in zip(n, self.state_labels):
            assigned_indices = np.random.choice(len(unassigned_nodes), num, replace=False)
            assigned_nodes = [unassigned_nodes[index] for index in assigned_indices]
            for node in assigned_nodes:
                self.node_states[node] = label
            assigned_nodes = set(assigned_nodes)
            unassigned_nodes = [node for node in unassigned_nodes if node not in assigned_nodes]

    def run(self):
        """
        Run the RDS simulation for t sweeps.
        """
        for sweep in range(1, self.t):
            # create a copy of the current node states to update
            new_node_states = self.node_states.copy()

            # iterate over randomly selected nodes in the graph
            nodes = list(self.G.nodes())
            for node_index in np.random.permutation(len(nodes)):
                node = nodes[node_index]
                # get the current state of the node
                current_state = self.node_states[node]

                # get the neighbors of the node
                neighbours = list(self.G.neighbors(node))

                # count neighbors in each state
                neighbour_counts = {label: 0 for label in self.state_labels}

                for neighbour in neighbours:
                    neighbour_counts[self.node_states[neighbour]] += 1

                # calculate new state of the current node
                new_state = self.get_new_state(current_state, neighbour_counts)
                new_node_states[node] = new_state

            # update the node states for this sweep
            self.node_states = new_node_states

            # update the state fractions for this sweep
            self.state_fracs[sweep] = [
                sum(state == label for state in self.node_states.values()) / self.N
                for label in self.state_labels
            ]

            # check if there are zero spreaders remaining
            if sum(state == 'S2' for state in self.node_states.values()) == 0:
                # truncate the state_fracs array to the current sweep and break the simulation
                self.state_fracs = self.state_fracs[:sweep + 1]
                break

    def get_new_state(self, current_state, neighbour_counts):
        """
        Calculate the new state of a node based on its current state and the states of its neighbours.
        
        Parameters
        ----------
        current_state: str
            the current state of the node
        neighbour_counts: dict
            a dictionary containing the number of neighbours in each state

        Returns
        -------
        new_state: str
            the new state of the node
        """
        # calculate all transition probabilities
        if current_state == 'S1':
            # transition probabilities for S1
            p12 = 1 - (1 - self.beta[0]) ** neighbour_counts['S2']
            p14 = 1 - (1 - self.beta[2]) ** neighbour_counts['S2']
            p15 = 1 - (1 - self.beta[3]) ** neighbour_counts['S3']
            p13 = 1 - (1 - self.beta[4]) ** neighbour_counts['S3']
            p11 = 1 - p12 - p14 - p15 - p13
            # choose new state based on probabilities
            new_state = np.random.choice(['S1', 'S2', 'S3', 'S4', 'S5'], p=[p11, p12, p13, p14, p15])
            return new_state
        elif current_state == 'S2':
            # transition probabilities for S2
            p24 = self.beta[1]
            p23 = 1 - (1 - self.beta[6]) ** neighbour_counts['S3']
            p22 = 1 - p24 - p23
            # choose new state based on probabilities
            new_state = np.random.choice(['S2', 'S3', 'S4'], p=[p22, p23, p24])
            return new_state
        elif current_state == 'S3':
            # transition probabilities for S3
            p35 = self.beta[5]
            p33 = 1 - p35
            # choose new state based on probabilities
            new_state = np.random.choice(['S3', 'S5'], p=[p33, p35])
            return new_state
        elif current_state == 'S4':
            # transition probabilities for S4
            p43 = 1 - (1 - self.beta[7]) ** neighbour_counts['S3']
            p45 = 1 - (1 - self.beta[8]) ** neighbour_counts['S3']
            p44 = 1 - p43 - p45
            # choose new state based on probabilities
            new_state = np.random.choice(['S4', 'S3', 'S5'], p=[p44, p43, p45])
            return new_state
        elif current_state == 'S5':
            # S5 is an absorbing state
            return 'S5'
    
