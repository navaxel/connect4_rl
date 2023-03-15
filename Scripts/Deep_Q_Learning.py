import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque
import copy
from tqdm.notebook import tqdm
import random

from kaggle_environments import make

import gym

import matplotlib.pyplot as plt
cols = 7
rows = 6

# Here we define some hyperparameters and the agent's architecture

observation_space = gym.spaces.Discrete(cols * rows).n
action_space = gym.spaces.Discrete(cols)


EPISODES = 1000
LR = 0.0001
MEM_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.95
EXPLORATION_MAX = 0.1
EXPLORATION_DECAY = 1.0
EXPLORATION_MIN = 0.1
sync_freq = 10

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (cols*rows,)
        self.action_space = action_space.n

        self.fc1 = nn.Linear(*self.input_shape, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.loss = nn.MSELoss()
        #self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class NetworkB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1536, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.loss = nn.MSELoss()
        
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layers(x)
        #x = x.mean(dim=1)
        return x
class ReplayBuffer:
    def __init__(self):
        self.memory = deque(maxlen=MEM_SIZE)
    
    def add(self, experience):
        self.memory.append(experience)
    
    def sample(self):
        minibatch = random.sample(self.memory, BATCH_SIZE)

        #print("len minibatch : ", len(minibatch))

        state1_batch = torch.stack([s1 for (s1,a,r,s2,d) in minibatch])
        action_batch = torch.tensor([a for (s1,a,r,s2,d) in minibatch])
        reward_batch = torch.tensor([r for (s1,a,r,s2,d) in minibatch])
        state2_batch = torch.stack([s2 for (s1,a,r,s2,d) in minibatch])
        done_batch = torch.tensor([d for (s1,a,r,s2,d) in minibatch])

        return (state1_batch, action_batch, reward_batch, state2_batch, done_batch)
class DQN:
    def __init__(self, network=NetworkB()):
        self.replay = ReplayBuffer()
        self.exploration_rate = EXPLORATION_MAX
        self.network = network
        self.network2 = copy.deepcopy(self.network) #A
        self.network2.load_state_dict(self.network.state_dict())


    def choose_action(self, observation,test = False):
        
        # Convert observation to PyTorch Tensor
        state = torch.tensor(observation).float().detach()
        #state = state.to(DEVICE)
        state = state.unsqueeze(0)
        possible_actions = list(np.where(state.numpy()[0][0][0] == 0)[0])
        
        if random.random() < self.exploration_rate and not test:
            random.choice(possible_actions)
   
        ### BEGIN SOLUTION ###

        # Get Q(s,.)
        
        q_values = self.network(state)

        # Choose the action to play    
        
        
        # action = torch.argmax(q_values).item()
        # return action 
        
        ### END SOLUTION ###

        action = torch.argmax(q_values[0,possible_actions]).item()
        return int(possible_actions[action])


    def learn(self):
        if len(self.replay.memory)< BATCH_SIZE:
            return

        ### BEGIN SOLUTION ###

        # Sample minibatch s1, a1, r1, s1', done_1, ... , sn, an, rn, sn', done_n
        state1_batch, action_batch, reward_batch, state2_batch, done_batch = self.replay.sample()

        # Compute Q values
        
        q_values = self.network(state1_batch)#.squeeze()
        
        with torch.no_grad():
            # Compute next Q values
            next_q_values = self.network2(state2_batch).squeeze()

        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        
        predicted_value_of_now = q_values[batch_indices, action_batch]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]

        # Compute the q_target
        q_target = reward_batch + GAMMA * predicted_value_of_future * (1-(done_batch).long())

        # Compute the loss (c.f. self.network.loss())
        loss = self.network.loss(q_target, predicted_value_of_now)

        ### END SOLUTION ###

        # Complute 𝛁Q
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


    def returning_epsilon(self):
        return self.exploration_rate
