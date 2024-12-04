import matplotlib.pyplot as plt
import numpy as np
img_folder = "img"
filename_learning = f"{img_folder}/TLC_naive_dqn.png"
filename_waiting = f"{img_folder}/waiting_time.png"
filename_avg_reward = f"{img_folder}/avg_reward.png"
filename_epsilon_decay = f"{img_folder}/epsilon_decay.png"
filename_throughput = f"{img_folder}/avg_throughput.png"

def plot_imgs(n_games,y,steps_array,eps_history,avg_wait_times_history,avg_rewards_history,avg_throughput):
    plot_learning_curve(steps_array, y, eps_history, filename_learning)
    plot_history_times(n_games,avg_wait_times_history,filename_waiting,"Average Wait Time","Average Wait Time per Episode")
    plot_history_times(n_games,avg_rewards_history,filename_avg_reward,"Average Reward", "Average Reward per Episode")
    plot_history_times(n_games,eps_history,filename_epsilon_decay,"Epsilon Decay","Epsilon Decay per Episode")
    plot_history_times(n_games,avg_throughput,filename_throughput,"Average Throughput","Average Throughput per Episode")


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    print("Saving plot...")
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="blue")
    ax.set_xlabel("Training Steps", color="blue")
    ax.set_ylabel("Epsilon", color="blue")
    ax.tick_params(axis='x', colors="blue")
    ax.tick_params(axis='y', colors="blue")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.plot(x, running_avg, color="red")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="red")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="red")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

def plot_history_times(x,avg_wait,filename,label,title):
    # print(avg_wait)
    # print(x)
    print("Saving plot...")
    plt.figure()
    plt.plot(range(x), avg_wait, label='Average Wait Time')
    plt.xlabel('Episodes')
    plt.ylabel(label)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)

def read_file(compare_type,norl):
    results_dir=""
    if norl:
        results_dir = f"./src/includes/sumo/{compare_type}/results/sumo.txt"
    else:
        results_dir = f"./src/includes/sumo/{compare_type}/results/rl.txt"
    history = []
    with open(results_dir, "r") as file:
        content = file.read()
        history = eval(content)
    
    return history
def plot_compare(compare_type):
    sumo = read_file(compare_type,True)
    rl = read_file(compare_type,False)

    x = range(len(sumo))
    # Creazione del grafico
    # Creazione del grafico
    plt.figure(figsize=(8, 5))
    plt.plot(x, sumo, label="SUMO", color='blue')
    plt.plot(x, rl, label="RL",  color='green')
    # Gestione della scala
    plt.xlim(250,450)  # Limiti per l'asse X
    # Personalizzazione del grafico
    plt.title(f"SUMO vs RL - {compare_type}", fontsize=14)
    plt.ylabel("Avg Wait Time", fontsize=12)
    plt.xlabel("Episodes", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Mostrare il grafico
    # plt.plot()
    plt.savefig(f"{img_folder}/{compare_type}/compare.png")
