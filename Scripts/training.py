import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from Scripts.Deep_Q_Learning import NetworkB,DQN

def train_agent(env, agent_to_train, agent_to_play_against, n_player=1, epochs=3000, rows=6, cols=7, sync_freq=10, display_info=False, save=False, path_to_save="", name="model_trained"):
    env.reset()

    if(n_player ==1):
        trainer = env.train([None,agent_to_play_against])
    else:
        trainer = env.train([agent_to_play_against,None])

    nb_win = 0
    score = 0

    for i in range(1, epochs):
        state, info = env.reset()
        state = state["observation"]["board"]
        state = np.reshape(state, [1, rows, cols])
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
                if reward == 1:
                    nb_win +=1
                if i%10==0 and display_info :
                    print("Episode {} Average Reward {} Nb Win {} Epsilon {}".format(i, score/i, nb_win, agent_to_train.returning_epsilon()))
                    nb_win = 0
                break

        if save:
            save_agent(agent_to_train, path_to_save, name, epochs)
            
    return agent_to_train


def train_adversial_agent(env, agent_to_train, agent_to_play_against, n_player=1, epochs=3000, rows=6, cols=7, sync_freq=10, display_info=False, save=False, path_to_save="", name="model_trained"):
    env.reset()

    nb_win = 0
    score = 0

    for i in range(1, epochs):
        state, info = env.reset()
        state = state["observation"]["board"]
        state = np.reshape(state, [1, rows, cols])
        
        j = 0

        if n_player == 2:
            first_action = agent_to_play_against.choose_action(state)
            state = np.reshape(env.step([first_action, None])[0]["observation"]["board"], [1, rows, cols])

        while True:
            j+=1
            
            trained_action = agent_to_train.choose_action(state)
            if n_player == 1:
                intermediary_state = np.reshape(env.step([trained_action, None])[0]["observation"]["board"], [1, rows, cols])
            else :
                intermediary_state = np.reshape(env.step([trained_action, None])[0]["observation"]["board"], [1, rows, cols])
                
            if not env.done:
                action_against = agent_to_play_against.choose_action(intermediary_state)
                if n_player == 1:
                    after_step = env.step([None, action_against])[0]
                else:
                    after_step = env.step([action_against, None])[0]
               
            
            state_after_action, reward, done, info = env.state[0]['observation']['board'], env.state[0]['reward'], env.done, env.state[0]['info'] 
            
            state_after_action = np.reshape(state_after_action, [1, rows, cols])
            state_after_action = torch.tensor(state_after_action).float()
            
            state = torch.tensor(state).float()


            exp = (state, trained_action, reward, state_after_action, done)
            agent_to_train.replay.add(exp)
            agent_to_train.learn()

            state = state_after_action
            score += reward

            if j % sync_freq == 0:
                agent_to_train.network2.load_state_dict(agent_to_train.network.state_dict())

            if done:
                if reward == 1:
                    nb_win += 1
                if i%10==0 and display_info :
                    print("Episode {} Average Reward {} Nb Win {} Epsilon {}".format(i, score/i, nb_win, agent_to_train.returning_epsilon()))
                    nb_win = 0
                break

        if save:
            save_agent(agent_to_train, path_to_save, name, epochs)
            
    return agent_to_train



def save_agent(agent, path_to_save, name, epochs):
    torch.save({'model_state_dict': agent.network.state_dict(),
        'optimizer_state_dict': agent.network.optimizer.state_dict(),
        'epoch': epochs}, path_to_save + name + ".pt")
    return None

def load_agent(PATH_TO_LOAD):
    state_dict = torch.load(PATH_TO_LOAD)
    model = NetworkB()
    model.load_state_dict(state_dict=state_dict["model_state_dict"])
    new_agent = DQN(network=model)
    return new_agent

