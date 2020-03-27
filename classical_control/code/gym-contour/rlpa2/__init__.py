from gym.envs.registration import register

register(
    'chakra-v0',
    entry_point='rlpa2.chakra:chakra',
)
register(
    'vishamC-v0',
    entry_point='rlpa2.vishamC:vishamC',
)