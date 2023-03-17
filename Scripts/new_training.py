import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from Scripts.Deep_Q_Learning import NetworkB,DQN

discount_factor = 0.9

def train_adversial_agent(env, agent_to_train, agent_to_play_against, n_player=1, epochs=3000, rows=6, cols=7, sync_freq=10, display_info=False, save=False, path_to_save="", name="model_trained"):
    env.reset()

    nb_win = 0
    score = 0

    for i in range(1, epochs):
        game_states = []
 
        
        state, info = env.reset()
        state = state["observation"]["board"]
        state = np.reshape(state, [1, rows, cols])
        
        j = 0

        if n_player == 2:
            first_action = agent_to_play_against.choose_action(1, state)
            state = np.reshape(env.step([first_action, None])[0]["observation"]["board"], [1, rows, cols])

        while True:
            j+=1
            
            trained_action = agent_to_train.choose_action(n_player, state)
            if n_player == 1:
                intermediary_state = np.reshape(env.step([trained_action, None])[0]["observation"]["board"], [1, rows, cols])
            else :
                intermediary_state = np.reshape(env.step([None, trained_action])[0]["observation"]["board"], [1, rows, cols])
            
            game_states.append(torch.tensor(intermediary_state).float())

            if not env.done:
                
                if n_player == 1:
                    action_against = agent_to_play_against.choose_action(2, intermediary_state)
                    
                    after_step = env.step([None, action_against])[0]
                else:
                    action_against = agent_to_play_against.choose_action(1, intermediary_state)
                    after_step = env.step([action_against, None])[0]

            state_after_action, reward, done, info = env.state[0]['observation']['board'], env.state[0]['reward'], env.done, env.state[0]['info'] 
            state_after_action = np.reshape(state_after_action, [1, rows, cols])
         
     

            state = state_after_action

            #if j % sync_freq == 0:
                #agent_to_train.network2.load_state_dict(agent_to_train.network.state_dict())

            if done:
                new_game_states = []
                game_discounted_rewards = []
                for state in reversed(game_states):
                    game_discounted_rewards.append(reward)
                    reward *= discount_factor
                    new_game_states.append(state)
                    
                
                agent_to_train.network.optimizer.zero_grad()
                predicted_output = agent_to_train.network(torch.stack(new_game_states))

                loss = agent_to_train.network.loss(predicted_output, torch.unsqueeze(torch.tensor(game_discounted_rewards),1))
                loss.backward()
                agent_to_train.network.optimizer.step()


                
                score += reward
                if reward > 0:
                    nb_win += 1
                if i%500==0 and display_info :
                    print("Episode {} Average Reward {} Nb Win {} Epsilon {}".format(i, score/i, nb_win, agent_to_train.returning_epsilon()))
           
                break



        if save:
            save_agent(agent_to_train, path_to_save, name, epochs)
            
    return agent_to_train

def train_agent(env, agent_to_train, agent_to_play_against, n_player=1, epochs=3000, rows=6, cols=7, sync_freq=10, display_info=False, save=False, path_to_save="", name="model_trained"):
    env.reset()

    nb_win = 0
    score = 0

    for i in range(1, epochs):
        game_states = []
 
        
        state, info = env.reset()
        state = state["observation"]["board"]
        state = np.reshape(state, [1, rows, cols])
        
        j = 0

        if n_player == 2:
            first_action = agent_to_play_against(env.state[0].observation,env.configuration)
            state = np.reshape(env.step([first_action, None])[0]["observation"]["board"], [1, rows, cols])

        while True:
            j+=1
            
            trained_action = agent_to_train.choose_action(n_player, state)
            if n_player == 1:
                intermediary_state = np.reshape(env.step([trained_action, None])[0]["observation"]["board"], [1, rows, cols])
            else :
                intermediary_state = np.reshape(env.step([None, trained_action])[0]["observation"]["board"], [1, rows, cols])
            
            game_states.append(torch.tensor(intermediary_state).float())

            if not env.done:
                
                if n_player == 1:
                    action_against =  agent_to_play_against(env.state[0].observation,env.configuration)
                    
                    after_step = env.step([None, action_against])[0]
                else:
                    action_against =  agent_to_play_against(env.state[0].observation,env.configuration)
                    after_step = env.step([action_against, None])[0]

            state_after_action, reward, done, info = env.state[0]['observation']['board'], env.state[0]['reward'], env.done, env.state[0]['info'] 
            state_after_action = np.reshape(state_after_action, [1, rows, cols])
         
     

            state = state_after_action

            #if j % sync_freq == 0:
                #agent_to_train.network2.load_state_dict(agent_to_train.network.state_dict())

            if done:
                new_game_states = []
                game_discounted_rewards = []
                for state in reversed(game_states):
                    game_discounted_rewards.append(reward)
                    reward *= discount_factor
                    new_game_states.append(state)
                    
                
                agent_to_train.network.optimizer.zero_grad()
                predicted_output = agent_to_train.network(torch.stack(new_game_states))

                loss = agent_to_train.network.loss(predicted_output, torch.unsqueeze(torch.tensor(game_discounted_rewards),1))
                loss.backward()
                agent_to_train.network.optimizer.step()


                
                score += reward
                if reward > 0:
                    nb_win += 1
                if i%500==0 and display_info :
                    print("Episode {} Average Reward {} Nb Win {} Epsilon {}".format(i, score/i, nb_win, agent_to_train.returning_epsilon()))
           
                break



        if save:
            save_agent(agent_to_train, path_to_save, name, epochs)
            
    return agent_to_train


def save_agent(agent, path_to_save, name, epochs):
    torch.save({'model_state_dict': agent.network.state_dict(),
        'optimizer_state_dict': agent.network.optimizer.state_dict(),
        'epoch': epochs}, path_to_save + name + ".pt")


def load_agent(PATH_TO_LOAD):
    state_dict = torch.load(PATH_TO_LOAD)
    model = NetworkB()
    model.load_state_dict(state_dict=state_dict["model_state_dict"])
    new_agent = DQN(network=model)
    return new_agent