from gym.envs.registration import register

register(
    id='puddleA-v0',
    entry_point='gym_puddle.envs:PuddleEnvA',
)
register(
    id='puddleB-v0',
    entry_point='gym_puddle.envs:PuddleEnvB',
)
register(
    id='puddleC-v0',
    entry_point='gym_puddle.envs:PuddleEnvC',
)