from kaggle_environments import make
import numpy as np
import gym
import torch
from Scripts.MCTS import agent_mcts
from Scripts.Deep_Q_Learning import DQN
import matplotlib.pyplot as plt
cols = 7
rows = 6
env = make("connectx", configuration={"rows":rows, "columns":cols})
sync_freq = 10
EPISODES = 100
observation_space = gym.spaces.Discrete(cols * rows).n
action_space = gym.spaces.Discrete(cols)
agent = DQN()

trainer = env.train([None, agent_mcts])

best_reward = 0
average_reward = 0
episode_number = []
average_reward_number = []

j=0
for i in range(1, EPISODES):
    state, info = env.reset()
    state = state["observation"]["board"]
    state = np.reshape(state, [1, observation_space])
    score = 0

    while True:
        j+=1

        action = agent.choose_action(state)
        state_, reward, done, info = trainer.step(action)
        if reward == None :
            reward = -10
        state_ = state_["board"]
        state_ = np.reshape(state_, [1, observation_space])
        state = torch.tensor(state).float()
        state_ = torch.tensor(state_).float()

        exp = (state, action, reward, state_, done)
        agent.replay.add(exp)
        agent.learn()
        
        state = state_
        score += reward

        if j % sync_freq == 0:
            agent.network2.load_state_dict(agent.network.state_dict())

        if done:
            if score > best_reward:
                best_reward = score
            average_reward += score 
            if i%10==0:
                print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}".format(i, average_reward/i, best_reward, score, agent.returning_epsilon()))
                #test_model(agent,10, observation_space)
            break
  
        episode_number.append(i)
        average_reward_number.append(average_reward/i)

torch.save({'model_state_dict': agent.network.state_dict(),
    'optimizer_state_dict': agent.network.optimizer.state_dict(),
    'epoch': EPISODES},"../models/"
    )