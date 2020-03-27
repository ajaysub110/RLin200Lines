## Soft Actor-Critic PyTorch Implementation

PyTorch implementation of the Soft Actor-Critic algorithm (without separate value function network) by Haarnoja et al.

### Running experiments
- Clone using ```git clone https://github.com/ajaysub110/sac-pytorch.git```
- Change hyperparameters in hyp.py file
- Train model using ```python main.py```

### Results

**Note**: For hyperparameters used for training, please refer [araffin/rl-baselines-zoo](https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/sac.yml)

### Episode reward vs Episode number Plots

1. **Pendulum-v0**

![Pendulum-v0](https://github.com/ajaysub110/sac-pytorch/blob/master/plots/Pendulum-v0.png)

2. **LunarLanderContinuous-v2**

![LunarLanderContinuous-v2](https://github.com/ajaysub110/sac-pytorch/blob/master/plots/LunarLander-v2.png)

3. **HopperBulletEnv-v0**

![HopperBulletEnv-v0](https://github.com/ajaysub110/sac-pytorch/blob/master/plots/hopperbullet_entropy_coeff_0-2.png)

4. **HalfCheetahBulletEnv-v0**

![HalfCheetahBulletEnv-v0](https://github.com/ajaysub110/sac-pytorch/blob/master/plots/halfcheetahbullet_auto_entropy.png)

5. **AntBulletEnv-v0**

![AntBulletEnv-v0](https://github.com/ajaysub110/sac-pytorch/blob/master/plots/antbullet_auto_entropy_more_episodes.png)

### Requirements
- PyTorch
- OpenAI Gym
- mujoco-py
- TensorBoard

### References
- https://arxiv.org/abs/1801.01290
- https://github.com/araffin/rl-baselines-zoo
- https://spinningup.openai.com/en/latest/algorithms/sac.html
- https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665
