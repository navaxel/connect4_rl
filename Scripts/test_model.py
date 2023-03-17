import numpy as np


def test_agent(env,agent_to_test,agent_against,n_player = 1,nb_games = 100,rows = 6,cols = 7):
    history = {1:0, -1:0, None:0, 0:0}
    exploration_rate = agent_to_test.exploration_rate
    agent_to_test.exploration_rate = 0
    observation, info = env.reset()
    observation = observation["observation"]

    for i in range(1, nb_games):
        state, info = env.reset()
        state = state["observation"]["board"]
        state = np.reshape(state, [1, rows, cols])
        
        j = 0

        if n_player == 2:
            first_action = agent_against(env.configuration, env.observation)
            state = np.reshape(env.step([first_action, None])[0]["observation"]["board"], [1, rows, cols])

        while True:
            j+=1
            
            trained_action = agent_to_test.choose_action(n_player, state, env)
            if n_player == 1:
                intermediary_state = np.reshape(env.step([trained_action, None])[0]["observation"]["board"], [1, rows, cols])
            else :
                intermediary_state = np.reshape(env.step([None, trained_action])[0]["observation"]["board"], [1, rows, cols])
                
            if not env.done:
                
                if n_player == 1:
                    action_against = agent_against(env.configuration, env.observation)
                    after_step = env.step([None, action_against])[0]
                else:
                    action_against = agent_against(env.configuration, env.observation)
                    after_step = env.step([action_against, None])[0]
               
            
            state_after_action, reward, done, info = env.state[0]['observation']['board'], env.state[0]['reward'], env.done, env.state[0]['info'] 
            
            state_after_action = np.reshape(state_after_action, [1, rows, cols])

            state = state_after_action

            if done:
                
                history[reward] += 1
            
                break

    agent_to_test.exploration_rate = exploration_rate
    
    return history

"""Final version of testing against an existing policy
"""
def new_testing(env,agent_to_test,agent_against,n_player = 1,nb_games = 100,rows = 6,cols = 7):
    history = {1:0, -1:0, None:0, 0:0}
    exploration_rate = agent_to_test.exploration_rate
    agent_to_test.exploration_rate = 0
    observation, info = env.reset()
    observation = observation["observation"]

    for i in range(1, nb_games):
        state, info = env.reset()
        state = state["observation"]["board"]
        state = np.reshape(state, [1, rows, cols])
        
        j = 0

        if n_player == 2:
            first_action = agent_against(env.state[0].observation,env.configuration)
            state = np.reshape(env.step([first_action, None])[0]["observation"]["board"], [1, rows, cols])

        while True:
            j+=1
            
            trained_action = agent_to_test.choose_action(n_player, state)
            if n_player == 1:
                intermediary_state = np.reshape(env.step([trained_action, None])[0]["observation"]["board"], [1, rows, cols])
            else :
                intermediary_state = np.reshape(env.step([None, trained_action])[0]["observation"]["board"], [1, rows, cols])
                
            if not env.done:
                
                if n_player == 1:

                    action_against = agent_against(env.state[0].observation,env.configuration)
                    after_step = env.step([None, action_against])[0]
                else:
                    action_against = agent_against(env.state[0].observation,env.configuration)
                    after_step = env.step([action_against, None])[0]
               
            
            state_after_action, reward, done, info = env.state[0]['observation']['board'], env.state[0]['reward'], env.done, env.state[0]['info'] 
            
            state_after_action = np.reshape(state_after_action, [1, rows, cols])

            state = state_after_action

            if done:
                
                history[reward] += 1
            
                break

    agent_to_test.exploration_rate = exploration_rate
    
    return history