import pickle
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
import math as Math
from typing import Callable, Dict
from imitation.algorithms.dagger import LinearBetaSchedule
from sklearn import tree
import numpy as np
import gym
from explainable.utils import collect_transitions, evaluate_policy


def toInt(value, max):
    newVal = 0
    if Math.floor(value)!=0:
        if value%Math.floor(value) >= 0.5:
            newVal= Math.ceil(value)
        else:
            newVal= Math.floor(value)
    else:
        if value >= 0.5:
            newVal= Math.ceil(value)
        else:
            newVal= Math.floor(value)
    if newVal<0:
        return 0
    else: 
        return (min(newVal,max))


class DAgger_Policy(BasePolicy):
    def __init__(self, teacher, student, beta_shedule, observation_space, action_space, **kwargs):
        super(BasePolicy, self).__init__(observation_space, action_space)
        self.student = student
        self.teacher = teacher
        self.train_mode = False
        self.beta_shedule = beta_shedule

    def forward(self, *args, **kwargs):
        pass
    
    def _predict(self, observation, deterministic: bool = False):
        pass
        
    def predict(
        self,
        observation,
        state = None,
        deterministic: bool = False,
        episode_start=0
    ):
        if self.train_mode==True and np.random.uniform(0, 1) < self.beta_shedule:
            if isinstance(self.teacher, BaseAlgorithm):
                return self.teacher.predict(observation[0],deterministic=True), state
            elif isinstance(self.teacher, Callable):
                return self.teacher(observation), None
            else:
                raise Exception("expert policy must be instance of BaseAlgorithm or a Callable") 
        else:
            return (toInt(self.student.predict(observation), self.action_space.n),None), state
    
    def save(self, output_dir, name='model.h5'):
        import os
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}{name}', 'wb') as f:
            pickle.dump(self.student, f)
    
    @staticmethod
    def load(dir, observation_space, action_space, teacher=None,  initial_beta = 150):
        with open(dir, 'rb') as f:
            policy = pickle.load(f)
            beta = LinearBetaSchedule(initial_beta)
            return DAgger_Policy(teacher, policy, beta.__call__(0),
                                        observation_space,
                                        action_space)
    

class DAgger():
    def __init__(self, teacher, student, env, demostrations, initial_beta:int=150,demos_per_beta=500, max_depth=8, min_impurity_decrease = 0.0, min_samples_split = 2):
        self.student = student.fit(demostrations['obs'],demostrations['acts']),
        self.beta = LinearBetaSchedule(initial_beta)
        self.initial_beta = initial_beta
        self.policy = DAgger_Policy(teacher, student, self.beta.__call__(0),
                                    env.observation_space,
                                    env.action_space
                                    )
        self.demos:Dict[str,list] = demostrations
        self.demos_per_beta = demos_per_beta
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = min_samples_split
        
    def train(self, teacher, env):
        self.policy.train_mode = True
        for count in range(Math.floor(4/3*self.initial_beta)):
            self.policy.beta_shedule = self.beta.__call__(count)
            trajectories = collect_transitions(
                self.policy,
                env,
                self.demos_per_beta
            )
            
            traj = trajectories['next_obs']
            
            np.random.shuffle(traj)
            
            traj = np.array(traj)
            
            if isinstance(teacher, BaseAlgorithm):
                acts, _ = teacher.predict(traj)
            elif isinstance(teacher, Callable):
                acts, _ = teacher(traj), None
            else:
                raise Exception("expert policy must be instance of BaseAlgorithm or a Callable") 
            
            self.demos['obs'].extend(trajectories['next_obs'])
            self.demos['acts'].extend(acts)
            self.student = tree.DecisionTreeClassifier(max_depth=self.max_depth, min_impurity_decrease=self.min_impurity_decrease, min_samples_split=self.min_samples_split)
            self.student.fit(self.demos['obs'],self.demos['acts'])
            self.policy = DAgger_Policy(teacher, self.student, self.beta.__call__(count),
                                    env.observation_space,
                                    env.action_space
                                    )
        self.policy.train_mode = False
        
    