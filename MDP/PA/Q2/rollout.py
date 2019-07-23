#!/usr/bin/env python

import click
import numpy as np
import gym
import rlpa2
import time

def include_bias(ob):
    return np.append(ob,1.0)

def get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    return rng.normal(loc=mean, scale=1.)

def get_grad_log_pi(ob,a,theta):
    ob_1 = np.reshape(include_bias(ob),(3,1))
    a = np.reshape(a,(2,1))
    return 0.5*np.dot((a-np.dot(theta,ob_1)),ob_1.T)
    

@click.command()
@click.argument("env_id", type=str, default="chakra")
def main(env_id):
    # Register the environment
    rng = np.random.RandomState(42)

    if env_id == 'chakra':
        env = gym.make('chakra-v0')
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    elif env_id == 'vishamC':
        env = gym.make('vishamC-v0')
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment: must be 'chakra' or 'vishamC'")

    env.seed(42)

    # Initialize parameters
    theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))
    max_itr = 400
    N = 5
    T = 100
    gamma = 0.9
    learning_rate = 0.001

    for i in range(max_itr): # for each iteration
        grad = 0
        reached = 0
        for j in range(N): # for each trajectory
            # Reset environment
            ob = env.reset()
            done = False
            # env.render()
            t_rewards = []
            t_obs = []
            t_actions = []
            # Generate a trajectory
            for k in range(T): # for each timestep
                action = get_action(theta,ob)
                next_ob,rew,done,_ = env.step(action)
                # env.render()
                t_obs.append(ob)
                t_actions.append(action)
                t_rewards.append(rew)
                ob = next_ob 
                if done:
                    reached += 1
                    break
            t_returns = [rew]
            for k in range(T-2,-1,-1):
                t_returns.append(t_rewards[k]+gamma*t_returns[-1])
            t_returns.reverse()
            grad_log = 0
            for k in range(T):
                grad_log += get_grad_log_pi(t_obs[k],t_actions[k],theta)
            grad += np.sum(t_returns)*grad_log
            print("Iteration: {}, Trajectory: {}, reward: {}".format(i,j,np.sum(t_rewards)))            
        grad = grad / N
        grad = grad / (np.linalg.norm(grad) + 1e-8)
        theta = theta - learning_rate * grad
        print("Iteration: {}, reached: {}, theta: {}\n".format(i,reached,theta))
    # env.viewer.close()

if __name__ == "__main__":
    main()
