# Introduction

The main objetive of this project is to apply Multi Agent Reinforcement Learning methods to solve the Continuous Control environment. 
The environment is described below:

The simulation contains 2 agents running in parallel on a collaborative-competitive setting.  At each time step, each agent has to choose a action for moving over x,z plane, delimited by the Tennis net. To achieve this, the action-space corresponds to the continuous increments in position from the Tennis net and jumping, bounded between the range [-1,1].


The state space has `8` variables corresponding to the position and velocity of the ball and racket.


Each agent has its own local observations
During the game, if an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The algorithm selected in this project was a Multi Agent Deep Deterministic Policy Gradient (MADDPG) agent.

# Implementation and Results

## The model

    The model architecture is a simple feedforward architecture, cascaded  over each fully connected layer. The snippet shown below describes the network architecture:

**Actor Network:**
```
    fc1 dimension : (8, 205)
    fc2 dimension : (205,152)
    fc3 dimension : (152,2)

    The forward structure:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
```

**Critic Network:**
```
    fc1 dimension : (426, 205)
    fc2 dimension : (205,152)
    fc3 dimension : (152,1)

    The forward structure:
        xs = torch.cat((state, action), dim=1)
        x = F.relu(self.fcs1(xs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

##  Hyperparametes 
 
 The table below sumarizes the network hyperparameters and structure

**Actor Network:**

| Layer             | Input Size | Output Size | Activation function |
|-------------------|------------|-------------|---------------------|
| Input layer (fc1) |      8     |     205     |         ReLU        |
|        fc2        |     205    |     152     |         ReLU        |
|        fc3        |     152    |      2      |         tanh        |

**Critic Network:**

|     **Layer**     | **Input Size** | **Output Size** | **Activation function** |
|:-----------------:|:--------------:|:---------------:|:-----------------------:|
| Input layer (fc1) |       426      |       205       |        ReLU       |
|        fc2        |       205      |       152       |        ReLU       |
|        fc3        |       152      |        1       |           linear          |

The linear output provided by the critic network is the clipped between [-1,1].

## Training  (TODO)

The training consists in a  simple loop structure. The learning agent interacts with the environment `n_episodes` times, until each of the episodes end, which occurs when the environments return the `done` flag. The episodes are limited in 5000 steps. In case the training isn't over until the end of the 5000 episodes, its considered as timeout.

The agent interacts and learns at each step, using the  Multi Agent DDPG  algorithm.

The exploration strategy is implemented as Ornstein-Uhlenbeck noise process, summed with the selected action. The agent noise  decay with time and there's a weight decay in the critic's loss function. So, at the end of the training, the actions are much like determined by the network outputs.

To get more statistically relevant scoring results, the analysed score is composed by a mean a moving average with the the number of samples corresponding to `WINDOW_SIZE`.

The most relevant parameters used are described in the table below:

| **Training Parameter** | **value** |
|:----------------------:|:---------:|
|    episode_length      |    999    |
|       n_agents         |     2     |
|      max_episodes      |    5000   |
|  SCORE_STOP_CONDITION  |     0.5   |
|       WINDOW_SIZE      |    100    |

| **Multi DDPG Agent Parameter** | **value** |
|:------------------------:|:---------:|
|        BUFFER_SIZE       |    10^6   |
|        BATCH_SIZE        |    256    |
|           GAMMA          |    0.98   |
|            TAU           |    1e-3   |
|         LR_ACTOR         |   1.2e-4  |
|         LR_CRITIC        |   1.5e-3  |
|       WEIGHT_DECAY       |    0.0    |
|        mu (noise)        |     0     |
|       theta (noise)      |    0.15   |
|       sigma (noise)      |    0.2    |


## Final scoring and Model benchmarking

The agents were able to achieve a mean score of 0.5 after 1878 episodes. Considering the WINDOW_SIZE, the environment was solved in a total of 1778 episodes

The graphic below shows the evolution of the agent score over the episodes, averaged between the moving average of the mean score of the 2 running agents.

<p align="center">
<object data="docs/average_scores_result.png" width="300" height="300"> </object>
</p>

The weights of the trained agent are stored in the models/episode-1878.pt file.

## Future work

    The project presented in this report is still very naive, compared to the state of the art of multi-agent learning methods. As future improvements, it can be done the following:

    - Improve the rigor of statistical analysis and improve the quality of metrics used;
    - Implement the AlphaZero learning algorithm to compare its performance with the results achieved by MADDPG for this task;
    - Improve and tune the network architecture and other hyperparameters such as learning rate, batch size, tau, etc;
    - Replace the noise function by a better suited process for this task;
    - Finish implementation of Prioritized Experience Replay (PER);
    - Train the agent on the Soccer environment, which is a harder task to learn.