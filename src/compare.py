from includes.sumo import SimEnv
import time
import argparse
from includes.DQAgent import DQAgent
from includes.Agent import Agent
import traci
from includes.utils import plot_compare
semaphores = ["gneJ26", "gneJ27", "gneJ28", "gneJ29", "gneJ30"]
def print_on_file(history,compare_type,norl):
    results_dir=""
    if norl:
        results_dir = f"./src/includes/sumo/{compare_type}/results/sumo.txt"
    else:
        results_dir = f"./src/includes/sumo/{compare_type}/results/rl.txt"
    
    with open(results_dir, "w") as file:
        file.write(str(history))
    
def simulate_with_rl(env, agents, semaphores, n_games, delay_time, compare_type):
    """Esegue la simulazione con RL."""
    observations = {}
    avg_wait_times_history = []
    decision_interval = 10  # Prendi decisioni ogni 5 passi

    # Inizializza le osservazioni iniziali per tutti i semafori
    for sem in agents.keys():
        obs, _, _ = agents[sem].step(0, 0)  # Azione random iniziale
        observations[sem] = obs


    # Loop principale della simulazione
    for i in range(n_games):
        total_wait_time = 0
        total_queue_length = 0
        total_score = 0

        # Prendi decisioni solo ogni `decision_interval` passi
        if i % decision_interval == 0:
            for sem, agent in agents.items():
                action = agent.choose_action(observations[sem])  # Azione basata sul modello
                observation_, reward, info = agent.step(action, i)

                # print(action)
                # Aggiorna metriche
                total_wait_time += info["avg_wait_time"]
                total_queue_length += info["queue_length"]
                total_score += reward

                # Aggiorna l'osservazione
                observations[sem] = observation_

            # Stampa metriche
            avg_wait_time = total_wait_time / len(semaphores)
            avg_queue_length = total_queue_length / len(semaphores)
            print(f"Step {i} | Avg Wait Time: {avg_wait_time:.2f}, Avg Queue Length: {avg_queue_length:.2f}")
            avg_wait_times_history.append(avg_wait_time)



        # Simula i passi intermedi
        env.ArrivedVehicles()
        # for _ in range(decision_interval):
            
        env.simulationStep()
        time.sleep(delay_time)
    print_on_file(avg_wait_times_history,compare_type,False)
    
    print(sum(avg_wait_times_history))


def simulate_without_rl(env, agents, n_games, delay_time, compare_type):
    """Esegue la simulazione senza RL."""
    avg_wait_times_history = []  
    decision_interval = 10

    for i in range(n_games):
        total_wait_time = 0

        if i % decision_interval == 0:
            for sem, agent in agents.items():
                    # Aggiorna metriche
                    agent.reset_lane_traffic_info_params()
                    total_wait_time += agent.info()["avg_wait_time"]

                # Stampa metriche
        
            avg_wait_time = total_wait_time / len(semaphores)
            print(f"Step {i} | Avg Wait Time:  {avg_wait_time:.2f}")
            avg_wait_times_history.append(avg_wait_time)

        env.simulationStep()
        time.sleep(delay_time)
    
    print_on_file(avg_wait_times_history,compare_type,True)
    
    print(sum(avg_wait_times_history))

def main(args):
    
    compare_type = args.type
    n_games = 10000 if compare_type == "curriculum" else args.steps

    delay_time = 0
    actions = 3 if compare_type == "random" else 2
    env = SimEnv(args, compare=True)
    env.start_sumo()

    if not args.norl:
        agents = {}
        for sem in semaphores:
            agents[sem] = DQAgent(gamma=0.99, epsilon=0, lr=0.0001,
                                  input_dims=8, n_actions=actions, mem_size=50000,
                                  batch_size=32, replace=1000, eps_dec=1e-5,
                                  chkpt_dir=f'./src/models/2x2', algo=f'{compare_type}',
                                  env_name=f'SUMO_tlc_2x2', TLC_name=sem)

            agents[sem].load_models()

        simulate_with_rl(env, agents, semaphores, n_games, delay_time, compare_type)
    else:
        agents = {}
        for sem in semaphores:
            agents[sem] = Agent(TLC_name=sem)
            traci.trafficlight.setProgram(sem, "0")
        simulate_without_rl(env, agents, n_games,  delay_time, compare_type)

    print(f"Throughput: {env.getArrivedVehicles()} vehicles arrived out of {env.getDepartedVehicles()} departed.")
    env.close_sumo()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DAI_RL - Traffic Light Control Simulation")
    parser.add_argument('--plot',  action="store_true", default=False, help='Run simulation without RL')
    parser.add_argument('--norl',  action="store_true", default=False, help='Run simulation without RL')
    parser.add_argument('--nogui',  action="store_true", default=False, help='Disable SUMO GUI')
    parser.add_argument('--steps', type=int, default=5000, help='Number of simulation steps')
    parser.add_argument('--type', type=str, required=True, help='Type [wave,curriculum,random]')
    args = parser.parse_args()
    if args.plot:
        plot_compare(args.type)
    else:
        if args.type in ["random","curriculum","wave"]:
            main(args)
        else:
            print("Type must be: [\"random\",\"curriculum\",\"wave\"]")
