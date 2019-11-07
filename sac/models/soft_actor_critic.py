import torch
import torch.optim as opt
import torch.nn.functional as F
import numpy as np

from .q_network import QNetwork
from .policy_network import PolicyNetwork
from helper import ReplayMemory, copy_params, soft_update
import hyp

class SoftActorCritic(object):
    def __init__(self,observation_space,action_space):
        self.s_dim = observation_space.shape[0]
        self.a_dim = action_space.shape[0]
        self.alpha = hyp.ALPHA

        # create component networks
        self.q_network_1 = QNetwork(self.s_dim,self.a_dim,hyp.H_DIM).to(hyp.device)
        self.q_network_2 = QNetwork(self.s_dim,self.a_dim,hyp.H_DIM).to(hyp.device)
        self.target_q_network_1 = QNetwork(self.s_dim,self.a_dim,hyp.H_DIM).to(hyp.device)
        self.target_q_network_2 = QNetwork(self.s_dim,self.a_dim,hyp.H_DIM).to(hyp.device)
        self.policy_network = PolicyNetwork(self.s_dim, self.a_dim, hyp.H_DIM, action_space).to(hyp.device)

        # copy weights from q networks to target networks
        copy_params(self.target_q_network_1, self.q_network_1)
        copy_params(self.target_q_network_2, self.q_network_2)
        
        # optimizers
        self.q_network_1_opt = opt.Adam(self.q_network_1.parameters(),hyp.LR)
        self.q_network_2_opt = opt.Adam(self.q_network_2.parameters(),hyp.LR)
        self.policy_network_opt = opt.Adam(self.policy_network.parameters(),hyp.LR)
        
        # automatic entropy tuning
        if hyp.ENTROPY_TUNING:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(hyp.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=hyp.device)
            self.alpha_optim = opt.Adam([self.log_alpha], lr=hyp.LR)
                
        self.replay_memory = ReplayMemory(hyp.REPLAY_MEMORY_SIZE)

    def get_action(self, s):
        state = torch.FloatTensor(s).to(hyp.device).unsqueeze(0)
        action, _, _ = self.policy_network.sample_action(state)
        return action.detach().cpu().numpy()[0]

    def update_params(self):
        states, actions, rewards, next_states, ndones = self.replay_memory.sample(hyp.BATCH_SIZE)
        
        # make sure all are torch tensors
        states = torch.FloatTensor(states).to(hyp.device)
        actions = torch.FloatTensor(actions).to(hyp.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(hyp.device)
        next_states = torch.FloatTensor(next_states).to(hyp.device)
        ndones = torch.FloatTensor(np.float32(ndones)).unsqueeze(1).to(hyp.device)

        # compute targets
        with torch.no_grad():
            next_action, next_log_pi,_ = self.policy_network.sample_action(next_states)
            next_target_q1 = self.target_q_network_1(next_states,next_action)
            next_target_q2 = self.target_q_network_2(next_states,next_action)
            next_target_q = torch.min(next_target_q1,next_target_q2) - self.alpha*next_log_pi
            next_q = rewards + hyp.GAMMA*ndones*next_target_q

        # compute losses
        q1 = self.q_network_1(states,actions)
        q2 = self.q_network_2(states,actions)

        q1_loss = F.mse_loss(q1,next_q)
        q2_loss = F.mse_loss(q2,next_q)
        
        pi, log_pi,_ = self.policy_network.sample_action(states)
        q1_pi = self.q_network_1(states,pi)
        q2_pi = self.q_network_2(states,pi)
        min_q_pi = torch.min(q1_pi,q2_pi)

        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        # gradient descent
        self.q_network_1_opt.zero_grad()
        q1_loss.backward()
        self.q_network_1_opt.step()

        self.q_network_2_opt.zero_grad()
        q2_loss.backward()
        self.q_network_2_opt.step()

        self.policy_network_opt.zero_grad()
        policy_loss.backward()
        self.policy_network_opt.step()

        # alpha loss
        if hyp.ENTROPY_TUNING:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(hyp.device)

        # update target network params
        soft_update(self.target_q_network_1,self.q_network_1)
        soft_update(self.target_q_network_2,self.q_network_2)

        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item()