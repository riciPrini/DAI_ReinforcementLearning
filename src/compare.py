from includes.sumo import SimEnv
import time
import argparse
from includes.DQAgent import DQAgent
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


def simulate_without_rl(env, n_games, delay_time):
    """Esegue la simulazione senza RL."""
    total_wait_time = {sem: 0 for sem in semaphores}  # Tempo totale di attesa per ogni semaforo
    total_queue_length = {sem: 0 for sem in semaphores}  # Lunghezza totale delle code per ogni semaforo
    num_vehicles_processed = {sem: 0 for sem in semaphores}  # Veicoli processati per semaforo
    avg_wait_times_history = []  # Storico tempi medi di attesa
    num_steps = 0

    for i in range(n_games):
        step_wait_time = {sem: 0 for sem in semaphores}  # Tempo di attesa per passo
        step_queue_length = {sem: 0 for sem in semaphores}  # Lunghezza delle code per pass

        # Stampa i risultati per ogni semaforo
        for sem in semaphores:
            # Ottieni la lista di tutti i veicoli nella simulazione
            for vehicle_id in env.getIDList():
                vehicle_wait_time = env.getAccumulatedWaitingTime(vehicle_id)
                vehicle_speed = env.getSpeed(vehicle_id)

            # Identifica il semaforo associato al veicolo
                controlling_tls = env.getVehicleControlTLS(vehicle_id)  # Trova il TLS che controlla il veicolo
                if controlling_tls and controlling_tls in semaphores:
                    step_wait_time[controlling_tls] += vehicle_wait_time
                    num_vehicles_processed[controlling_tls] += 1

                # Considera un veicolo in coda se la velocità è molto bassa
                    if vehicle_speed < 0.1:  # Velocità considerata come fermo
                        step_queue_length[controlling_tls] += 1
            
            # Aggiorna le metriche totali per ogni semaforo
            total_wait_time[sem] += step_wait_time[sem]
            total_queue_length[sem] += step_queue_length[sem]
            if num_vehicles_processed[sem] > 0:
                avg_wait_time = total_wait_time[sem] / num_vehicles_processed[sem]
             
        avg_wait_time = total_wait_time[sem] / num_steps if num_steps > 0 else 0
        avg_queue_length = total_queue_length[sem] / num_steps if num_steps > 0 else 0
        print(f"Episode {i} Avg Wait Time: {avg_wait_time:.2f}, "
                  f"Avg Queue Length: {avg_queue_length:.2f}, Vehicles Processed: {num_vehicles_processed[sem]}")

        avg_wait_times_history.append(avg_wait_time)   
        num_steps += 1

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
                                  chkpt_dir=f'.\src\models\{grid}', algo='DQNAgent',
                                  env_name=f'SUMO_tlc_{grid}', TLC_name=sem)

            agents[sem].load_models()

        simulate_with_rl(env, agents, semaphores, n_games, delay_time)
    else:
        simulate_without_rl(env, n_games, delay_time)

    print(f"Throughput: {env.getArrivedVehicles()} vehicles arrived out of {env.getDepartedVehicles()} departed.")
    env.close_sumo()

if __name__ == '__main__':
    main()
