from gym.envs.registration import register

register(
    id='RMSA-v0',
    entry_point='explainable.envs:RMSAEnv',
)

register(
    id='DeepRMSA-v0',
    entry_point='explainable.envs:DeepRMSAEnv',
)

register(
    id='DeepRMSAKSP-v0',
    entry_point='explainable.envs:DeepRMSAKSPEnv',
)