# Report

## Introduction

In this project our goal is to train an agent having two jointed arm in Unity's Reacher environment so that it could be in contact with its target for as long as possible.
According to the environment, **a reward of +0.1 is received each time step the agent is in contact with its target**. The environment is considered solved when the agent is able
to achieve a score of more than or equal to +30 on average over a period of 100 episodes.

The simulated environment provides a simplified **state space having 33 dimensions** corresponding to the agent's position, rotation, velocity, and angular velocities of the arm. **There are four actions in continuous domain forming a vector with four numbers each representing torque applicable to two joints :**

## Learning Techniques

Since, the environment consists of 33 dimensions having continuous values as state space, also has four actions which are **continuous in nature** and we need to solve the problem in model free environment because dynamics of the environment are not pre knowledge. It is not possible to use Q learning or any value based learning approach directly as these cannot be used to handle continuous action spaces efficiently.
Thus, we have two options we could use, either we could use Policy based methods such as Proximal policy optimization or Actor Critic methods.

In this project we choose to use DDPG (Deep deterministic policy gradient) algorithm which works like Actor Critic method and can be regarded as a varient of DQN algorithm which can handle even continuous action spaces efficient in a model free environment. The research paper explaining this method can be found [here](https://arxiv.org/abs/1509.02971).

### About the Algorithm

The DDPG algorithm maintains a parametrized actor function ![Alt Text](images/actor_symbol.PNG) which specifies the current policy by deterministically mapping states to a specific action. The Critic Q(s,a) is learned using the Bellman equation as in Q-learning.
The actor is updated by applying the chain rule to the expected return from the start distribution J with respect to the actor parameters :

![Alt Text](images/ddpg_learning.PNG)

#### Use of Replay buffer

As in DQN algorithm, one of the challenge in off policy learning algorithms learning samples should be independently and identically distributed. When the samples are generated from exploring sequentially in an environment this assumption no longer holds. Also, to make efficient use of hardware optimizations, it is essential to learn in mini-batches, rather than online.
To solve this issue, as in DQN, this algorithm uses replay buffer to address this issue. Transitions are sampled from the environment according to the exploration policy and the tuple (s, a, r, st+1) is stored in replay buffer which is a cache of finite size. When the replay buffer is full the oldest samples are discarded. At each timestep the actor and critic are
updated by sampling a minibatch uniformly from the buffer.

#### Seperation of active and target networks

When active network being updated is also used in calculating the target value, the Q update is prone to divergence. The solution in this algorithm is similar to the target network used in DQN algorithm, but modified for actor critic using "soft" target updates, rather than directly copying the weights. Actor and Critic networks are initially copied to form target actor 
and Critic networks, the weights of these target networks are updated separately and in slow manner making use of a hyperparameter. Thus, the target networks are constrained to change slowly, greatly improving the stability of learning.

#### Use of batch normalization

When learning from low dimensional feature vector observations, the different components of the observation may have different physical units and the ranges may vary across environments. This can make it difficult for the network to learn effectively and may make it difficult to find hyper-parameters.
In this algorithm, this issue is addressed by adapting a recent technique in deep learning called batch normalization. This technique normalizes each dimension across the samples in a minibatch to have a unit mean and variance. In addition, it maintains a running average of the mean and variance to use for normalization during testing. Batch normalization is used on state input and all layers of the actor network and critic network.

#### Exploration technique

A major challenge in learning in continuous action spaces is exploration. An advantage of off policy algorithms such as DDPG is that we can treat the problem of exploration independently from the learning algorithm. Here, exploration policy is constructed by adding noise sampled from a process N to actor policy. Ornstein-Uhlenbeck process has been chosen as Noise process.

![Alt Text](images/noise.PNG)

#### DDPG Algorithm

The DDPG algorithm as specified in the paper is shown below for convenience :

![Alt Text](images/ddpg_algorithm.PNG)

## Neural Network Architecture

In this project two neural networks are used one for actor and one for critic. 
Actor is created using a feed forward network with two hidden layers having 400 and 300 neurons each. Batch normalization layers are used after both the hidden layer for the purpose of batch normalization. ReLu is used as activation for each layer except the output layer which uses tanh as activation function.
Critic is created using a feed forward network with two hidden layers having 400 and 300 neurons each. Batch normalization layers are used here too after both the hidden layer. ReLu is used as activation function for all hidden layers.

## Hyper-parameters

The important hyper-parameter values which we have used are:
- BUFFER_SIZE = int(1e6): replay buffer cache size
- BATCH_SIZE = 256      : minibatch size which designates the amount of experience tuples which are extracted from Buffer for each step of learning
- GAMMA = 0.99          : discount factor
- TAU = 1e-3            : parameter to control the speed of soft update of target network parameters
- LR_ACTOR = 1e-3       : learning rate of the actor 
- LR_CRITIC = 1e-3      : learning rate of the critic
- UPDATE_EVERY = 20     : number of steps between every round of updates
- N_UPDATES = 10        : number of batches in a single round of updates

**Special important values are the UPDATE_EVERY and N_UPDATES hyper-parameters**. N_UPDATES designates how many times learning is iterated at each time step learning when it is decided to learn.
Since, N_UPDATES value is 10, this means that at each time step when learning is to be attempted, the exercise of taking batch of samples and learning from them is done 10 times before
moving to the next timestep.
UPDATE_EVERY designates after how many time steps should we attempt to do learning. It is observed that if we attempt to do learning at each time step then the learning results in much 
fluctuations and not able to converge.

Another important set of parameters which are necessary to be set properly are **parameters pertaining to Noise following Ornstein-Uhlenbeck process**. In this project, theta is set to 0.03 and sigma is set to 0.02. If higher values are set for these parameters
the actor is not able to train faster and converge because the noise factor becomes high, thus, it becomes difficult for the training algorithm to zero in to the optimal policy.

**Batch normalization turned out to be really helpful** Without batch normalizing layers the models were not able to converge faster. I observed that adding each normalization layer one by one resulted in significant improvement.

## Results

The agent converges and solves the environment in 163 Episodes. The Score vs Episodes plot of training is as shown below:

![Alt Text](images/results.PNG)

## Future improvements

- Instead of random selection of tuples from replay buffer use prioritized replay which could lead to even faster convergence.
- Try solving this environment using other techniques such as TRPO and PPO.
- Instead of just using one agent as done in this project, try solving using multi agent approach like done in A3C algorithm. This will allow us to use multi agent working parallel at the same time to extract samples thus avoiding the use of Replay Buffer and exploiting parallel processing and/or distributed computing.