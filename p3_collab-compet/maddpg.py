import torch
from ddpg_agent import DDPGAgent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from replay_buffer import ReplayBuffer
import numpy as np
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size

class MADDPG:

    def __init__(self, state_size,action_size, n_agents=1, random_seed=0):
        self.agents = [DDPGAgent(state_size,action_size,random_seed) for x in range(n_agents)]
        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, n_agents)
        self.n_agents = n_agents
        self.action_size = action_size
        self.state_size = state_size

    def step(self, states, actions, rewards, next_states, dones, priority):
        # TODO: Retrieve the priority from the TD error computed by the DDPG agent

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