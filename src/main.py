from includes.sumo import SimEnv
import time
import traci
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from includes.DQAgent import DQAgent
from includes.utils import plot_imgs


semaphores = ["gneJ26", "gneJ27", "gneJ28", "gneJ29", "gneJ30"]
def graph():
    G = nx.Graph()
    for sem in semaphores:
        G.add_node(sem)

    # Aggiungi connessioni (archi) nel grafo
    for sem in semaphores:
        for neighbor in get_neighbors(sem):
            G.add_edge(sem, neighbor)

    pos = nx.spring_layout(G)  # Layout del grafo
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.ion()  # Abilita la modalità interattiva per la visualizzazione dinamica
def get_neighbors(semaphore):
    neighbors = {
        "gneJ26": ["gneJ27","gneJ28","gneJ29","gneJ30"],  
        "gneJ27": ["gneJ26"],   
        "gneJ28": ["gneJ26"],   
        "gneJ29": ["gneJ26"],   
        "gneJ30": ["gneJ26"],   
    }
    return neighbors.get(semaphore, [])



if __name__ == '__main__':
    env = SimEnv()
    env.start_sumo()
        
    step = 0
    delayTime = 0 #1/8
    n_games = 100
    n_steps = 0
    agents = {} 
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
        queue_lengths = []
        for sem, agent in agents.items():
            neighbors = {}
            received_message = False
            # Get neighbor queues
            neighbor_queues = [agents[neighbor].get_queue_length() for neighbor in get_neighbors(sem)]
            # Update with new neigh info
            agent.update_with_neighbor_info(neighbor_queues)
            for neighbor in get_neighbors(sem):
                queue_length = agents[neighbor].get_queue_length()
                neighbors[neighbor] = queue_length 
                if queue_length > 0:
                    received_message = True

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
            
            # node_colors = []

            # for node in G.nodes():
            #     if node == sem and received_message:
            #         node_colors.append('red')  # Cambia colore a rosso se ha ricevuto info
            #     else:
            #         node_colors.append('blue')  # Altrimenti rimane blu
                
            # queue_lengths.append(sum(neighbor_queues))
            # # # Disegna il grafo durante ogni ciclo
            # ax.clear()
            # nx.draw(G, pos, with_labels=True, node_size=3000,
            #         node_color=node_colors, ax=ax)
            # ax.set_title(f"Simulation Step {i} | Semaphore: {sem}")
            # plt.draw()
            # plt.pause(0.1)  # Pausa breve per aggiornare la visualizzazione
            
            avg_score = np.mean(scores[sem][-100:]) #if sem in scores else 0

            ## DEBUG
            
            agent.print_neighbor_info(neighbors)
            

            if avg_score > best_score:
                if not load_checkpoint:
                    agent.save_models()
                best_score = avg_score
            
            scores[sem].append(score)
            steps_array.append(n_steps)
        

        #FIX
        avg_wait_time = total_wait_time / len(semaphores) if len(semaphores) > 0 else 0
        avg_queue_length = total_queue_length / len(semaphores) 

        avg_wait_times_history.append(avg_wait_time)
        avg_rewards_history.append(total_episode_reward)
        eps_history.append(np.mean([agent.epsilon for agent in agents.values()]))
        total_rewards.append(total_score)
      

        env.simulationStep()
        time.sleep(delayTime)
            # step+=1


   
    
    
    steps_array = [i+1 for i in range(n_games)]
    y = [np.mean([scores[sem][i] for sem in semaphores]) for i in range(len(scores[semaphores[0]]))]
    
    
    # plot_imgs(n_games,y,steps_array,eps_history,avg_wait_times_history,avg_rewards_history) # (Un)comment to plot
    

    env.close_sumo() 

