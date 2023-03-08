import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from Scripts.Deep_Q_Learning import NetworkB,DQN

def train_agent(env,agent_to_train,agent_to_play_against,n_player = 1,epoch = 3000,rows = 6, cols = 7,sync_freq = 10,display_info = False):
    env.reset()

    if(n_player ==1):
        trainer = env.train([None,agent_to_play_against])
    else:
        trainer = env.train([agent_to_play_against,None])

    for i in range(1, epoch):
        state, info = env.reset()
        state = state["observation"]["board"]
        state = np.reshape(state, [1, rows, cols])
        score = 0
        j = 0
        while True:
            j+=1

            action = agent_to_train.choose_action(state)
            state_, reward, done, info = trainer.step(action)
            if reward == None :
                reward = -1000
            state_ = state_["board"]
            state_ = np.reshape(state_, [1, rows, cols])
            state = torch.tensor(state).float()
            state_ = torch.tensor(state_).float()

            exp = (state, action, reward, state_, done)
            agent_to_train.replay.add(exp)
            agent_to_train.learn()
            state = state_
            score += reward
            if j % sync_freq == 0:
                agent_to_train.network2.load_state_dict(agent_to_train.network.state_dict())

            if done:
                
                average_reward1 = score 
                if i%10==0 and display_info :
                    print("Episode {} Average Reward {} Last Reward {} Epsilon {}".format(i, average_reward1/i,  score, agent_to_train.returning_epsilon()))
                break
            
    return agent_to_train

def save_agent(agent,PATH_TO_SAVE):
    return None

def load_agent(PATH_TO_LOAD):
    state_dict = torch.load(PATH_TO_LOAD)
    model = NetworkB()
    model.load_state_dict(state_dict=state_dict["model_state_dict"])
    new_agent = DQN(network=model)
    return new_agent

