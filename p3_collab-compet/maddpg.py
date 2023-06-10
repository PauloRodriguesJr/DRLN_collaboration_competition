# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

import torch
from ddpg_agent import DDPGAgent
from utilities import transpose_to_tensor, transpose_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
# device = 'cpu'

# agent = DDPGAgent(state_size=observation_space, action_size=action_space, n_parallel=1, random_seed=0)


class MADDPG:
    def __init__(self, state_size, action_size, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 14+2+2+2=20

        self.maddpg_agent = [DDPGAgent(state_size, action_size, n_parallel=1, random_seed=0),
                             DDPGAgent(state_size, action_size, n_parallel=1, random_seed=0)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    # def get_actors(self):
    #     """get actors of all the agents in the MADDPG object"""
    #     actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent] # ddpg_agent.actor
    #     return actors

    # def get_target_actors(self):
    #     """get target_actors of all the agents in the MADDPG object"""
    #     target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent] # ddpg_agent.target_actor
    #     return target_actors

    # INPUT: List of state observations
    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions


    def update(self, samples, agent_number, logger=None):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        obs , action, reward, next_obs, done = samples

        print(np.shape(obs))
        print(np.shape(action))
        print(np.shape(reward))
        print(np.shape(next_obs))
        print(np.shape(done))
        
        # obs_full = torch.stack(obs_full)
        # next_obs_full = torch.stack(next_obs_full)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network

        # # print(next_obs[agent_number].cpu().data.numpy())

        # print(next_obs)
        # print(type(next_obs))
        # print(len(next_obs))

        # print(next_obs[0])
        # print(type(next_obs[0]))
        # print(np.shape(next_obs[0]))
        # print(np.shape(next_obs))
        
        target_actions = self.target_act(next_obs) #cpu().data.numpy()
        
        print(target_actions)
        print(target_actions.shape)
        # target_actions = torch.cat(target_actions, dim=1)
        
        target_critic_input = np.hstack((next_obs,target_actions))

        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = reward[:,agent_number,:] + self.discount_factor * q_next * (1 - done[:,agent_number,:])
        action = torch.cat(action, dim=1)
        # critic_input = torch.cat((obs, action), dim=1).to(device)
        q = agent.critic(np.hstack((obs, action)))

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative

        #DUVIDA A RESPEITO DELE, SERÁ QUE FAZ SENTIDO?
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        q_input2 = torch.cat((obs.flatten().t(), q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        # logger.add_scalars('agent%i/losses' % agent_number,
                        #    {'critic loss': cl,
                            # 'actor_loss': al},
                        #    self.iter)
        return al, cl

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.soft_update(ddpg_agent.critic_local, ddpg_agent.critic_target, self.tau)
            ddpg_agent.soft_update(ddpg_agent.actor_local, ddpg_agent.actor_target, self.tau)



