import numpy as np

N = 10
n = np.array([2, 3, 1, 2, 2])

# define possible states
state_labels = ['I', 'S', 'D', 'Sr', 'Sd']

# populate the initial states of the nodes in the graph
node_states = np.empty(N, dtype='<U2')

for num, label in zip(n, state_labels):
    node_states[np.random.choice(np.where(node_states == '')[0], num, replace=False)] = label

print(node_states)