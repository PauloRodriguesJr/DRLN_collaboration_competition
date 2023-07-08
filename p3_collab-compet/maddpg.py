# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

import torch
from ddpg_agent import DDPGAgent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from replay_buffer import ReplayBuffer
import numpy as np
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size

# class MADDPG:
#     def __init__(self, state_size, action_size, episode_length, discount_factor=0.95, tau=0.02):
#         super(MADDPG, self).__init__()

#         # critic input = obs_full + actions = 14+2+2+2=20
#         self.buffer = ReplayBuffer(action_size, int(BUFFER_SIZE), BATCH_SIZE, seed=0, n_agents=2)

#         self.maddpg_agent = [DDPGAgent(state_size, action_size, self.buffer, random_seed=0, n_parallel=1),
#                              DDPGAgent(state_size, action_size, self.buffer,random_seed=0, n_parallel=1)]
        
#         self.discount_factor = discount_factor
#         self.tau = tau
#         self.iter = 0

#     def act(self, obs_all_agents, noise=0.0):
#         """get actions from all agents in the MADDPG object"""
#         actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
#         return actions

#     def target_act(self, obs_all_agents, noise=0.0):
#         """get target network actions from all the agents in the MADDPG object """
#         target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
#         return target_actions


#     def step(self, states, actions, rewards, next_states, dones):
#         self.buffer.add(states, actions, rewards, next_states, dones)
#         for ddpg_agent in self.maddpg_agent:
#             ddpg_agent.step()

#     def reset(self):        
#         for ddpg_agent in self.maddpg_agent:
#             ddpg_agent.reset()

#     def update_targets(self):
#         """soft update targets"""
#         self.iter += 1
#         for ddpg_agent in self.maddpg_agent:
#             ddpg_agent.soft_update(ddpg_agent.critic_local, ddpg_agent.critic_target, self.tau)
#             ddpg_agent.soft_update(ddpg_agent.actor_local, ddpg_agent.actor_target, self.tau)


### FOR DEBUGGING ###
class MADDPG:

    def __init__(self, state_size,action_size, n_agents=1, random_seed=0):
        self.agents = [DDPGAgent(state_size,action_size,random_seed) for x in range(n_agents)]
        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, n_agents)
        self.n_agents = n_agents
        self.action_size = action_size
        self.state_size = state_size

    def step(self, states, actions, rewards, next_states, dones, priority):
        self.buffer.add(states, actions, rewards, next_states, dones, priority)

        if len(self.buffer) > BATCH_SIZE:
            for agent in self.agents:
                agent.step(self.buffer.sample())

    def act(self, states, add_noise=True):
        actions = np.zeros([self.n_agents, self.action_size])
        for index, agent in enumerate(self.agents):
            actions[index, :] = agent.act(states[index], add_noise)
        return actions

    def reset(self):        
        for agent in self.agents:
            agent.reset()