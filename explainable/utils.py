import gym
import pandas as pd
import networkx as nx
import numpy as np
import warnings
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
from stable_baselines3.common import base_class
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Tuple, Union





class Path:
    def __init__(self, path_id, node_list, length, best_modulation=None):
        self.path_id = path_id
        self.node_list = node_list
        self.length = length
        self.best_modulation = best_modulation
        self.hops = len(node_list) - 1


class Service:
    def __init__(self, service_id, source, source_id, destination=None, destination_id=None, arrival_time=None,
                holding_time=None, bit_rate=None, best_modulation=None, service_class=None, number_slots=None):
        self.service_id = service_id
        self.arrival_time = arrival_time
        self.holding_time = holding_time
        self.source = source
        self.source_id = source_id
        self.destination = destination
        self.destination_id = destination_id
        self.bit_rate = bit_rate
        self.service_class = service_class
        self.best_modulation = best_modulation
        self.number_slots = number_slots
        self.number_slots_backup = number_slots
        self.route = None
        self.initial_slot = None
        self.backup_route = None
        self.initial_slot_backup = None
        self.accepted = False
        self.shared = False
        self.dpp = False

    def __str__(self):
        msg = '{'
        msg += '' if self.bit_rate is None else f'br: {self.bit_rate}, '
        msg += '' if self.service_class is None else f'cl: {self.service_class}, '
        return f'Serv. {self.service_id} ({self.source} -> {self.destination})' + msg

def start_environment(env, steps):
    done = True
    for i in range(steps):
        if done:
            env.reset()
        while not done:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
    return env


def get_k_shortest_paths(G, source, target, k, weight=None):
    """
    Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
    """
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def get_path_weight(graph, path, weight='length'):
    return np.sum([graph[path[i]][path[i + 1]][weight] for i in range(len(path) - 1)])


def random_policy(env):
    return env.action_space.sample()


def collect_transitions(
    teacher,
    enviroment,
    n_timesteps
):
    transitions = rollout.generate_transitions(
        teacher,
        DummyVecEnv([lambda: RolloutInfoWrapper(enviroment)]),
        n_timesteps,
        rng=np.random.default_rng()
    )
    traj = dict(obs = [], acts = [], next_obs = [])
    for rol in transitions:
        traj['obs'].append(rol['obs'])
        traj['acts'].append(rol['acts'])
        traj['next_obs'].append(rol['next_obs'])
    return traj


def evaluate_policy(
    env,
    n_eval_episodes: int = 10,
    model: Union["base_class.BaseAlgorithm", None] = None,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    warn: bool = True,
    return_episode_rewards: bool = False,
    heuristic_policy = None,
    return_dataframe = False,
    seed: int = 10
) -> Union[Tuple[float, float], Tuple[List[float], List[int]], Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The optical_network environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    # Avoid circular import
    from .envs.optical_network_env import OpticalNetworkEnv
    from stable_baselines3.common.monitor import Monitor
    
    is_monitor_wrapped = False

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])
    
    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    
    env.seed(seed)
    observations = env.reset()
    
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    df = {}
    while (episode_counts < episode_count_targets).any():
        
        if model is not None:
            actions, states = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
        else:
            actions = heuristic_policy(observations)
        
        # print(observations[0])
        # print('action',actions)
        observations, rewards, dones, infos = env.step(actions)
        # print('reward',rewards)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if return_dataframe:
                        for key in info.keys():
                            if key in df:
                                df[key].append(info[key])
                            else:
                                df[key]=[info[key]]
                    
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        if return_dataframe:
            return episode_rewards, episode_lengths, pd.DataFrame(df)
        return episode_rewards, episode_lengths
    else:
        if return_dataframe:
            return mean_reward, std_reward, pd.DataFrame(df)
        return mean_reward, std_reward


def linear_schedule(initial_value: float, final_value=0) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    
    
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        assert initial_value > final_value
        return final_value + progress_remaining * (initial_value - final_value)

    return func

