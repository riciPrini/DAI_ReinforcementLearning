from includes.sumo import SimEnv
import time
import traci
import argparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from includes.DQAgent import DQAgent
from includes.utils import plot_imgs
def get_neighbors(semaphore,grid):
    if grid == "2x2":
        neighbors = {
            "gneJ26": ["gneJ27","gneJ28","gneJ29","gneJ30"],  
            "gneJ27": ["gneJ26"],   
            "gneJ28": ["gneJ26"],   
            "gneJ29": ["gneJ26"],   
            "gneJ30": ["gneJ26"],   
        }
    elif grid == "3x3":
        neighbors = {
            "gneJ24": ["gneJ25","gneJ26"],  
            "gneJ25": ["gneJ24","gneJ27"],  
            "gneJ26": ["gneJ24","gneJ27","gneJ28"],  
            "gneJ27": ["gneJ25","gneJ26","gneJ29"],   
            "gneJ28": ["gneJ26","gneJ29","gneJ30"],   
            "gneJ29": ["gneJ27","gneJ28","gneJ31"],   
            "gneJ30": ["gneJ28","gneJ31"],   
            "gneJ31": ["gneJ29","gneJ30"],   
        }
        
    return neighbors.get(semaphore, [])
load_checkpoint = True  # Assicura che i modelli vengano caricati
compare = True
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DAI_RL - Traffic Light Control Simulation")
    parser.add_argument('--env', type=str, required=True, help='Enveriorment')    
    parser.add_argument('--nogui',  action="store_true", default=False, help='Sumo GUI')
    parser.add_argument('--steps', type=int, default=5000, help='Steps number')
    args = parser.parse_args()
    n_games = args.steps
    grid = args.env 
    step = 0
    delayTime = 0 #1/8
    agents = {} 
    if grid == "2x2":
        semaphores = ["gneJ26", "gneJ27", "gneJ28", "gneJ29", "gneJ30"]
    elif grid == "3x3":
        semaphores = ["gneJ24","gneJ25","gneJ26", "gneJ27", "gneJ28", "gneJ29", "gneJ30","gneJ31"]

    env = SimEnv(args,compare)
    env.start_sumo()
    # Inizializza gli agenti
    for sem in semaphores:
        agents[sem] = DQAgent(gamma=0.99, epsilon=0, lr=0.0001,
                            input_dims=8, n_actions=2, mem_size=50000,
                            batch_size=32, replace=1000, eps_dec=1e-5,
                            chkpt_dir=f'src/models/{grid}', algo='DQNAgent',
                            env_name=f'SUMO_tlc_{grid}', TLC_name=sem)

    if load_checkpoint:
        for sem, agent in agents.items():
            agent.load_models()  # Carica i modelli salvati

    observations = {}
    for sem in agents.keys():
        obs, _, _ = agents[sem].step(0, step)  # Azione random iniziale
        observations[sem] = obs
    for i in range(n_games):
        total_wait_time = 0
        total_queue_length = 0
        total_score = 0

        for sem, agent in agents.items():
            action = agent.choose_action(observations[sem])  # Azione basata sul modello    
            observation_, reward, info = agent.step(action, step)

            # Aggiorna metriche
            total_wait_time += info["avg_wait_time"]
            total_queue_length += info["queue_length"]
            total_score += reward

            observations[sem] = observation_  # Aggiorna osservazione

        env.ArrivedVehicles()
        # Calcola metriche
        avg_wait_time = total_wait_time / len(semaphores)
        avg_queue_length = total_queue_length / len(semaphores)
        print(f"Episode {i} | Avg Wait Time: {avg_wait_time:.2f}, Avg Queue Length: {avg_queue_length:.2f}")

        env.simulationStep()
        
        time.sleep(delayTime)

    print(f" {env.getArrivedVehicles()} throughput after simulation on {env.getDepartedVehicles()} departed vehicles")
    
    env.close_sumo()