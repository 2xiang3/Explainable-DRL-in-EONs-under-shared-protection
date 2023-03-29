from .utils import Path
from .graph_utils import read_sndlib_topology, read_txt_file, get_k_shortest_paths, get_path_weight



def _get_modulation_format(length, _modulations):
    for i in range(len(_modulations) - 1):
        if length > _modulations[i + 1]['maximum_length'] and length <= _modulations[i]['maximum_length']:
            return _modulations[i]
    return _modulations[len(_modulations) - 1]


def create_topology_from_file(file_dir, topology_name, modulations, k_paths=1):
    k_shortest_paths = {}
    if file_dir.endswith('.xml'):
        topology = read_sndlib_topology(file_dir)
    elif file_dir.endswith('.txt'):
        topology = read_txt_file(file_dir)
    else:
        raise ValueError('Supplied topology is unknown')
    idp = 0
    for idn1, n1 in enumerate(topology.nodes()):
        for idn2, n2 in enumerate(topology.nodes()):
            if idn1 < idn2:
                paths = get_k_shortest_paths(topology, n1, n2, k_paths, weight='length')
                print(n1, n2, len(paths))
                lengths = [get_path_weight(topology, path, weight='length') for path in paths]
                selected_modulations = [_get_modulation_format(length, modulations) for length in lengths]
                objs = []
                for path, length, modulation in zip(paths, lengths, selected_modulations):
                    objs.append(Path(idp, path, length, best_modulation=modulation))
                    print('\t', idp, length, modulation, path)
                    idp += 1
                k_shortest_paths[n1, n2] = objs
                k_shortest_paths[n2, n1] = objs
    topology.graph['name'] = topology_name
    topology.graph['ksp'] = k_shortest_paths
    topology.graph['modulations'] = modulations
    topology.graph['k_paths'] = k_paths
    topology.graph['node_indices'] = []
    for idx, node in enumerate(topology.nodes()):
        topology.graph['node_indices'].append(node)
        topology.nodes[node]['index'] = idx
    return topology

def generate_node_request_probabilities(num_nodes, seed=1):
    import numpy as np
    np.random.seed = seed
    probs = [np.random.exponential(size=num_nodes)]
    total = np.sum(probs)
    return [value/total for value in probs]
