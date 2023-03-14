import numpy as np
def test_agent(env,agent_to_test,agent_against,n_player = 1,nb_games = 100,rows = 6,cols = 7):

    exploration_rate = agent_to_test.exploration_rate
    agent_to_test.exploration_rate = 0
    observation, info = env.reset()
    observation = observation["observation"]

    i_games = 0
    history = {1:0, -1:0, None:0, 0:0}
    done = False

    if n_player ==1:
        trainer = env.train([None,agent_against])
    else:
        trainer = env.train([agent_against,None])
    while i_games < nb_games:
        env.render()
        
        if done :
            #if reward == None:
            #    print("reward : ", reward)
            #    print("board : ", np.reshape(observation["board"], [1, rows, cols]))
            
            observation, info = env.reset()
            observation = observation["observation"]
            history[reward] += 1
            i_games +=1
            
        action = agent_to_test.choose_action(np.reshape(observation["board"], [1, rows, cols]),test = True)
        observation, reward, done, info = trainer.step(action)

    agent_to_test.exploration_rate = exploration_rate
    
    return history