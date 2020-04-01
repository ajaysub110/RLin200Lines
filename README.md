# RL in 200 Lines
PyTorch implementations of Reinforcement Learning algorithms in less than 200 lines.

## Algorithms:

1. **Deep Reinforcement Learning**
    - DQN
    - Soft Actor-Critic (SAC) [[Results]](https://drive.google.com/open?id=1rrzC4DigBsKWXv9YVmV0jI1vhstnwFKd)
    - Vanilla Policy Gradient (Actor-Critic) [[Results]](https://drive.google.com/open?id=1T9rqRIfZcCe61h2Ib-Q9Fyf2A-B9th2A)
    - Proximal Policy Optimization (PPO) [[Results]](https://drive.google.com/open?id=1zb6bt5RSZUnCTRw8crWPypXHkiECyRG4)
    - Deep Deterministic Policy Gradient (DDPG) [[Results]](https://drive.google.com/open?id=1IS06f5od8-mNvi9oaSlIOqFZoxJpdNcx)

2. **Bandits**
    - Epsilon Greedy
    - Softmax action selection
    - UCB-1
    - REINFORCE

3. **Classical MDP Control**
    - SARSA
    - Q-learning
    - SARSA(lambda)
    - Vanilla Policy Gradient

4. **Additional Resources**
    - Report on Bandit algorithms
    - Report on Classical MDP control algorithms
    - Contour environment - *gym-contour* 
    - Puddle world - *gym-puddle*

## Dependencies
- PyTorch
- Tensorboard
- OpenAI Gym
- Numpy

## Usage
- Clone the repository.
- Run experiments on an algorithm by running either <name>.py or main.py within its directory.
- Tensorboard of my experiments can be viewed by using the 'Result' links given above.

## References

- **Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor**, (2018) [[bib]](./bibtex.bib#L9-L15)  by *Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel and Sergey Levine*

- **Proximal Policy Optimization Algorithms**, (2017) [[bib]](./bibtex.bib#L25-L31)  by *John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford and Oleg Klimov*

- **Benchmarking Deep Reinforcement Learning for Continuous Control**, (2016) [[bib]](./bibtex.bib#L17-L23)  by *Yan Duan, Xi Chen, Rein Houthooft, John Schulman and Pieter Abbeel*

- **Playing Atari with Deep Reinforcement Learning**, (2013) [[bib]](./bibtex.bib#L1-L7)  by *Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra and Martin A. Riedmiller*

- **Using Confidence Bounds for Exploitation-Exploration Trade-offs**, (2002) [[bib]](./bibtex.bib#L42-L49)  by *Peter Auer*

- **Eligibility Traces for Off-Policy Policy Evaluation**, (2000) [[bib]](./bibtex.bib#L60-L65)  by *Doina Precup, Richard S. Sutton and Satinder P. Singh*

- **Policy Gradient Methods for Reinforcement Learning with Function Approximation**, (1999) [[bib]](./bibtex.bib#L67-L72)  by *Richard S. Sutton, David A. McAllester, Satinder P. Singh and Yishay Mansour*

- **Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning**, (1992) [[bib]](./bibtex.bib#L33-L40)  by *Ronald J. Williams*

- **Q-learning**, (1992) [[bib]](./bibtex.bib#L51-L58)  by *Chris Watkins and Peter Dayan*

- **Deterministic Policy Gradient Algorithms**, (2014) [[bib]](../bibtex.bib#L74-L79)  by *David Silver, Guy Lever, Nicolas Manfred Otto Heess, Thomas Degris, Daan Wierstra and Martin A. Riedmiller*