from includes.sumo import SimEnv
import time
import argparse
from includes.DQAgent import DQAgent
from includes.Agent import Agent
from includes.utils import plot_compare
semaphores = ["gneJ26", "gneJ27", "gneJ28", "gneJ29", "gneJ30"]

def simulate_with_rl(env, agents, semaphores, n_games, delay_time):
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
    print(avg_wait_times_history)
    
    print(sum(avg_wait_times_history))


def simulate_without_rl(env, agents, n_games, delay_time):
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
    
    print(avg_wait_times_history)
    
    print(sum(avg_wait_times_history))

def main():
    parser = argparse.ArgumentParser(description="DAI_RL - Traffic Light Control Simulation")
    parser.add_argument('--norl',  action="store_true", default=False, help='Run simulation without RL')
    parser.add_argument('--nogui',  action="store_true", default=False, help='Disable SUMO GUI')
    parser.add_argument('--steps', type=int, default=5000, help='Number of simulation steps')
    args = parser.parse_args()

    grid = "2x2"
    n_games = args.steps
    delay_time = 0
    semaphores = ["gneJ26", "gneJ27", "gneJ28", "gneJ29", "gneJ30"] if grid == "2x2" else [
        "gneJ24", "gneJ25", "gneJ26", "gneJ27", "gneJ28", "gneJ29", "gneJ30", "gneJ31"
    ]

    env = SimEnv(args, compare=True)
    env.start_sumo()

    if not args.norl:
        agents = {}
        for sem in semaphores:
            agents[sem] = DQAgent(gamma=0.99, epsilon=0, lr=0.0001,
                                  input_dims=8, n_actions=2, mem_size=50000,
                                  batch_size=32, replace=1000, eps_dec=1e-5,
                                  chkpt_dir=f'./src/models/{grid}', algo='DQNAgent',
                                  env_name=f'SUMO_tlc', TLC_name=sem)

            agents[sem].load_models()

        simulate_with_rl(env, agents, semaphores, n_games, delay_time)
    else:
        agents = {}
        for sem in semaphores:
            agents[sem] = Agent(TLC_name=sem)
        simulate_without_rl(env, agents, n_games,  delay_time)

    print(f"Throughput: {env.getArrivedVehicles()} vehicles arrived out of {env.getDepartedVehicles()} departed.")
    env.close_sumo()
if __name__ == '__main__':
    plot_compare()
    # main()
