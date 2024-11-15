from includes.sumo import SimEnv
import time
import traci
import numpy as np
import matplotlib.pyplot as plt
from includes.DQAgent import DQAgent
from includes.utils import plot_learning_curve,plot_wait_times

def get_neighbors(semaphore):
    neighbors = {
        "gneJ26": ["gneJ27"],  
        "gneJ27": ["gneJ26"]   
    }
    return neighbors.get(semaphore, [])

if __name__ == '__main__':
    env = SimEnv()
    env.start_sumo()
        
    step = 0
    delayTime = 0 #1/8
    semaphores = ["gneJ26", "gneJ27"]
    # Central = "gneJ26"
    n_games = 5000
    n_steps = 0
    agents = {} 
    # scores = []
    scores = {sem: [] for sem in semaphores}
    eps_history = []
    avg_wait_times_history = []
    avg_rewards_history = []
    total_rewards = []
    steps_array = []

    best_score = -np.inf
    load_checkpoint = False

    for sem in semaphores:
        agents[sem] = DQAgent(gamma=0.99, epsilon=1, lr=0.0001,
                          input_dims=8, n_actions=2, mem_size=50000,
                          batch_size=32, replace=1000, eps_dec=1e-5,
                          chkpt_dir='models/', algo='DQNAgent',
                          env_name='SUMO_tlc', TLC_name=sem)

    if load_checkpoint:
        for sem, agent in agents.items():
            agent.load_models()
    
    observations = {}
    for sem in agents.keys():
        obs, _, _ = agents[sem].step(0, step)  # Azione random iniziale
        observations[sem] = obs
    # observation, reward, info = agent.step(0, step) #taking random action

    for i in range(n_games):
        total_score = 0 # total scores of all traffic_lights
        total_wait_time = 0  # Total wait time (for metrics purposes)
        total_queue_length = 0  # Total queue length
        total_episode_reward = 0 # rewards sum for each epsiode
        for sem, agent in agents.items():
            # Get neighbor queues
            neighbor_queues = [agents[neighbor].get_queue_length() for neighbor in get_neighbors(sem)]
            # Update with new neigh info
            agent.update_with_neighbor_info(neighbor_queues)

            score = 0
            action = agent.choose_action(observations[sem])
            observation_, reward, info = agent.step(action, step)
            score += reward

            # Metrics purposes 
            total_wait_time += info["avg_wait_time"]
            total_queue_length += info["queue_length"]

            if not load_checkpoint:
                agent.store_transition(observations[sem], action, reward, observation_)
                agent.learn()

            observations[sem] = observation_
            
            n_steps += 1
            total_score += score # accumulate multi-tl scores

            
            avg_score = np.mean(scores[sem][-100:]) #if sem in scores else 0

            ## DEBUG
            # agent.printStatusReport(step)
            print(f'Agent {sem} | Episode: {i}, Score: {score}, Avg Score: {avg_score:.1f}, Best Score: {best_score:.2f}, Epsilon: {agent.epsilon:.2f}, Steps: {n_steps}')
            # print('episode: ', i,'score: ', score,
            #      ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            #     'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

            if avg_score > best_score:
                if not load_checkpoint:
                    agent.save_models()
                best_score = avg_score
            
            scores[sem].append(score)
            steps_array.append(n_steps)

        
        avg_wait_time = total_wait_time / len(semaphores)
        avg_queue_length = total_queue_length / len(semaphores) 
        print(f"Episode {i} | Avg Wait Time: {avg_wait_time:.2f}, Avg Queue Length: {avg_queue_length:.2f}, Total Reward: {total_episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}")  
        avg_wait_times_history.append(avg_wait_time)
        avg_rewards_history.append(total_episode_reward)
        eps_history.append(np.mean([agent.epsilon for agent in agents.values()]))
        total_rewards.append(total_score)
        env.simulationStep()
        time.sleep(delayTime)
            # step+=1

  
    
    
    steps_array = [i+1 for i in range(n_games)]
    y = [np.mean([scores[sem][i] for sem in semaphores]) for i in range(len(scores[semaphores[0]]))]
    filename_learning = 'TLC_naive_dqn.png'
    filename_waiting = 'waiting_time.png'
    filename_avg_reward = 'avg_reward.png'
    filename_epsilon_decay = 'epsilon_decay.png'
    plot_learning_curve(steps_array, y, eps_history, filename_learning)
    plot_wait_times(n_games,avg_wait_times_history,filename_waiting)
    plot_wait_times(n_games,avg_rewards_history,filename_avg_reward)
    plot_wait_times(n_games,eps_history,filename_epsilon_decay)
    env.close_sumo() 

