

def generate_node_request_probabilities(num_nodes, seed=1):
    import numpy as np
    np.random.seed(seed)
    probs = [np.random.exponential(size=num_nodes)]
    total = np.sum(probs)
    return [value/total for value in probs]



print(generate_node_request_probabilities(46))