# Deep RL implementations

## Vanilla Policy Gradient
Value function baseline (Actor-Critic), Deep network policy and value function.

`runs/vpg_0`: No batch, no baseline
`runs/vpg_1`: No batch, value function baseline
`runs/vpg_2`: batch update, value function baseline

## Proximal Policy Optimization
Clipping loss, Deep network policy and value function

`runs/ppo_0`: With clipping loss approximation (from SpinningUp)
`runs/ppo_1`: With original clipping loss, from paper