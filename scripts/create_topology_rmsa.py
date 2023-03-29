
import pickle
import os
from explainable.topology import create_topology_from_file


# defining the EON parameters
# definitions according to : https://github.com/xiaoliangchenUCD/DeepRMSA/blob/eb2f2442acc25574e9efb4104ea245e9e05d9821/K-SP-FF%20benchmark_NSFNET.py#L268
_modulations = list()
# modulation: string description
# capacity: Gbps / slot
# maximum_distance: km
_modulations.append({'modulation': 'BPSK', 'capacity': 12.5, 'maximum_length': 100000})
_modulations.append({'modulation': 'QPSK', 'capacity': 25., 'maximum_length': 2000})
_modulations.append({'modulation': '8QAM', 'capacity': 37.5, 'maximum_length': 1250})
_modulations.append({'modulation': '16QAM', 'capacity': 50., 'maximum_length': 625})


k_paths=1
root_dir = os.path.abspath(os.curdir)
top_name = 'cost239'
outputdir = 'demo'
topology_dir = '/topologies/' +  top_name +'.txt'
topology = create_topology_from_file(root_dir + topology_dir,  top_name, _modulations, k_paths=k_paths)

with open(f'{root_dir}/topologies/'+ outputdir + '/' +  top_name +f'_{k_paths}.h5', 'wb') as f:
    pickle.dump(topology, f)

print('done for', topology)