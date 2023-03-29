import enum
from typing import Tuple
import gym
import numpy as np

from .rmsa_env import RMSAEnv
from .optical_network_env import OpticalNetworkEnv


class DeepRMSAEnv(RMSAEnv):
    
    def __init__(self, topology=None, j=1,
                    episode_length=1000,
                    mean_service_holding_time=25.0,
                    mean_service_inter_arrival_time=0.1,
                    num_spectrum_resources=1000,
                    node_request_probabilities=None,
                    seed=None,
                    k_paths=5,
                    allow_rejection=False,
                    only_spectrum_obs = False,
                    reward_function = None):
            super().__init__(topology=topology,
                            episode_length=episode_length,
                            load=mean_service_holding_time / mean_service_inter_arrival_time,
                            mean_service_holding_time=mean_service_holding_time,
                            num_spectrum_resources=num_spectrum_resources,
                            node_request_probabilities=node_request_probabilities,
                            seed=seed,
                            k_paths=k_paths,
                            allow_rejection=allow_rejection,
                            reset=False)

            self.j = j
            self.only_spectrum_obs = only_spectrum_obs
            self.reward_function = reward_function
            shape = (2 * self.j + 3) * self.k_paths if only_spectrum_obs else 1 + 2 * self.topology.number_of_nodes() \
                + (2 * self.j + 3) * self.k_paths * 2 #doubled
            self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.float32, shape=(shape,))
            self.action_space = gym.spaces.Discrete(self.k_paths * self.j * 2 + self.reject_action) #doubled
            self.action_space.seed(self.rand_seed)
            self.observation_space.seed(self.rand_seed)
            self.reset(only_counters=False)

    def step(self, action: int):
        if action < self.k_paths * self.j * 2:  # action is for assigning a path needs to be doubled or deleted (get_path_block_id takes it into consideration)
            # path, block = self._get_path_block_id(action)
            working_path, block_working, backup_path, block_backup = self._get_path_block_id(action)
            # print("------------------")
            # print("Working path: ", working_path)
            #Maybe add prints
            working_initial_indices, working_lengths = self.get_available_blocks_working(working_path)  #here
            # print("Initial indices working : {}; LengthsW :{} ".format(working_initial_indices, working_lengths))
            backup_initial_indices, backup_lengths = self.get_available_blocks_backup(backup_path) #here
            backup_dpp_initial_indices, backup_lengths_dpp = self.get_available_blocks_working(backup_path)
            # print("BU path: ", backup_path)
            # print("Initial indices BU : {}; LengthsB :{} ".format(backup_initial_indices, backup_lengths))
            if (block_working < len(working_initial_indices)) and (block_backup < len(backup_initial_indices)) and (block_backup < len(backup_dpp_initial_indices)): #shared and dedicated
                return super().step([working_path, working_initial_indices[block_working], backup_path,
                                     backup_initial_indices[block_backup], backup_path, backup_dpp_initial_indices[block_backup]])

            if ((block_working < len(working_initial_indices)) and (block_backup < len(backup_dpp_initial_indices))): #shared or dedicated only
                return super().step([working_path, working_initial_indices[block_working],  backup_path,                                    
                                      backup_dpp_initial_indices[block_backup],  backup_path, backup_dpp_initial_indices[block_backup]])

            if ((block_working < len(working_initial_indices)) and (block_backup < len(backup_initial_indices))): #shared or dedicated only
                return super().step([working_path, working_initial_indices[block_working], backup_path,
                                     backup_initial_indices[block_backup], backup_path, backup_initial_indices[block_backup]])

            else:
                return super().step([self.k_paths, self.num_spectrum_resources,self.k_paths, self.num_spectrum_resources,self.k_paths, self.num_spectrum_resources])
        else:
            return super().step([self.k_paths, self.num_spectrum_resources,self.k_paths, self.num_spectrum_resources,self.k_paths, self.num_spectrum_resources])  

    def observation(self):
        # observation space defined as in https://github.com/xiaoliangchenUCD/DeepRMSA/blob/eb2f2442acc25574e9efb4104ea245e9e05d9821/DeepRMSA_Agent.py#L384
        source_destination_tau = np.zeros((2, self.topology.number_of_nodes()))
        min_node = min(self.service.source_id, self.service.destination_id)
        max_node = max(self.service.source_id, self.service.destination_id)
        source_destination_tau[0, min_node] = 1
        source_destination_tau[1, max_node] = 1
        spectrum_obs = np.full((self.k_paths * 2, 2 * self.j + 3), fill_value=-1.)
        for idp, path in enumerate(self.k_shortest_paths[self.service.source, self.service.destination]):
            available_slots = self.get_available_slots_working(path)
            # demanda de FSUs
            num_slots = self.get_number_slots(path)
            initial_indices, lengths = self.get_available_blocks_working(idp)

            if len(initial_indices)>0:
                for idb, (initial_index, length) in enumerate(zip(initial_indices, lengths)):
                    # initial slot index
                    spectrum_obs[idp, idb * 2 + 0] = 2 * (initial_index - .5 * self.num_spectrum_resources) / self.num_spectrum_resources
                    # number of contiguous FS available
                    spectrum_obs[idp, idb * 2 + 1] = (length - 8) / 8
            else:
                for j in range(self.j):
                    spectrum_obs[idp, j * 2 + 0] = 1 # route is unavailable
                
            spectrum_obs[idp, self.j * 2] = (num_slots - 5.5) / 3.5 # number of FSs necessary
            idx, values, lengths = DeepRMSAEnv.rle(available_slots)
            spectrum_obs[idp, self.j * 2 + 1] = 2 * (np.sum(available_slots) - .5 * self.num_spectrum_resources) / self.num_spectrum_resources # total number available FSs
            av_indices = np.argwhere(values == 1) # getting indices which have value 1
            if av_indices.shape[0] > 0:
                spectrum_obs[idp, self.j * 2 + 2] = (np.mean(lengths[av_indices]) - 4) / 4 # avg. number of slot in the available blocks
        bit_rate_obs = np.zeros((1, 1))
        bit_rate_obs[0, 0] = self.service.bit_rate / 100

        if self.only_spectrum_obs:
            return spectrum_obs.reshape((1, np.prod(spectrum_obs.shape))).reshape(self.observation_space.shape)
        else:
            return np.concatenate((bit_rate_obs, source_destination_tau.reshape((1, np.prod(source_destination_tau.shape))),
                                spectrum_obs.reshape((1, np.prod(spectrum_obs.shape)))), axis=1)\
                .reshape(self.observation_space.shape)

    def reward(self):
        if self.reward_function is None:
            if self.service.shared:
                return 2

            if self.service.dpp:
                return 1
            
            if not self.service.accepted:
                return -1
        else:
            return self.reward_function()

    def reset(self, only_counters=True):
        super().reset(only_counters=only_counters)
        return self.observation()

    def _get_path_block_id(self, action: int) -> Tuple[int, int, int, int]: 
        
        working_path = action // (self.j * self.j * self.k_paths) % self.k_paths
        block_working = action // (self.j*self.k_paths) %self.j 
        backup_path = action % (self.j*self.k_paths) // self.j
        block_backup = action % self.j

        return working_path, block_working, backup_path, block_backup


def shortest_path_first_fit(env: DeepRMSAEnv) -> int:
    if not env.allow_rejection:
        return 0
    else:
        initial_indices, lengths = env.get_available_blocks(0)
        if len(initial_indices) > 0:  # if there are available slots
            return 0
        else:
            return env.k_paths * env.j


def shortest_available_path_first_fit(env: DeepRMSAEnv) -> int:
    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        initial_indices, lengths = env.get_available_blocks(idp)
        if len(initial_indices) > 0:  # if there are available slots
            return idp * env.j  # this path uses the first one
    return env.k_paths * env.j