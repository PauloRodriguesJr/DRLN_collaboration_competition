# Introduction

The main objetive of this project is to apply Multi Agent Reinforcement Learning methods to solve the Continuous Control environment. 
The environment is described below:

The simulation contains 2 agents running in parallel on a collaborative-competitive setting.  At each time step, each agent has to choose a continuous action for moving in x,z plane, delimited by the Tennis net. To achieve this movimentation, the action-space corresponds to the continuous increments in position and jump, bounded between the range [-1,1].


The state space has `33` variables, which are position, rotation, linear velocity and angular velocity for each of the robot's arm.

Each time step the agent keeps its arm on target position, a reward of +0.1 is gained. Thus, the goal of the environment is to keep the robot arm inside at the target as many steps as possible. 

The algorithm selected in this project was a DDPG agent.


# Implementation and Results

## The model

    The model architecture is cascaded except by the Advantage-Baseline part. The code snippet shown below describes the network:

```    num_neurons=48
    fc1 (37, 48)
    fc2 (48, 48)
    fc_advantage = (48, 4)
    fc_baseline = (48, 1)

    x = self.fc1(state)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    advantage = F.relu(self.fc_advantage(x))
    baseline = F.relu(self.fc_baseline(x))
```
##  Hyperparametes 
 
 The table below sumarizes the network hyperparameters and structure

Actor Network:

| Layer             | Input Size | Output Size | Activation function |
|-------------------|------------|-------------|---------------------|
| Input layer (fc1) |     33     |     256     |         ReLU        |
|        fc2        |     256    |      64     |         ReLU        |
|        fc3        |     64     |      4      |         tanh        |


Critic Network:

|     **Layer**     | **Input Size** | **Output Size** | **Activation function** |
|:-----------------:|:--------------:|:---------------:|:-----------------------:|
| Input layer (fc1) |       33       |       128       |        leaky ReLU       |
|        fc2        |       128      |       64        |        leaky ReLU       |
|        fc3        |       64       |        32       |           tanh          |
|        fc4        |       32       |        1        |            -            |


All layers are fully connected, and cascaded in sequence.


## Training 

The training consists in a  simple loop structure. In this structure, the learning agent interact with the environment`n_episodes`, until the each of the episodes end, which occurs when the environments return the `done` flag. The episodes contains 800 steps until the end, receiving rewards only when it's close to the target.

The agent interacts and learns at each step, using the DDPG  algorithm.

The exploration strategy is implemented as Ornstein-Uhlenbeck noise process, summed with the selected action. The agent noise doesn't decay with time, but there's a weight decay in a the critic's loss function.

To get more statistically relevant scoring results, the analysed score is composed by a mean a moving average with the the number of samples corresponding to `WINDOW_SIZE`.


The most relevant parameters used are described in the table below:


| **Training Parameter** | **value** |
|:----------------------:|:---------:|
|          t_max         |    1000   |
|       N_PARALLEL       |     20    |
|      max_episodes      |    800    |
|  SCORE_STOP_CONDITION  |     30    |
|       WINDOW_SIZE      |    100    |

| **DDPG Agent Parameter** | **value** |
|:------------------------:|:---------:|
|        BUFFER_SIZE       |    10^6   |
|        BATCH_SIZE        |    256    |
|           GAMMA          |    0.99   |
|            TAU           |    1e-3   |
|         LR_ACTOR         |    1e-4   |
|         LR_CRITIC        |    1e-3   |
|       WEIGHT_DECAY       |  0.00001  |
|        mu (noise)        |     0     |
|       sigma (noise)      |    0.2    |
|       theta (noise)      |    0.15   |



## Final scoring and Model benchmarking

The 20 agents were able to achieve a mean score of 30 after 202 episodes. Considering the WINDOW_SIZE, the environment was solved in a total of 102 episodes

The graphic below shows the evolution of the agent score over the episodes, averaged between the 20 running agents.

<p align="center">
<object data="docs/DDPG_learning_dynamics.png" width="300" height="300"> </object>
</p>

## Future work

    The project presented in this report is still very naive, compared to the state of the art of policy-based learning methods. As future improvements, it can be done the following:
    - Improve the statistical analysis relevance and metrics quality
    - Implement a D4PG or PPO and compare their performance with the results achieved by DDPG for this task.
    - Improve and tune the network architecture
    - Change the noise function
    - Train the agent on the Crawler environment, which is a harder task to learn

