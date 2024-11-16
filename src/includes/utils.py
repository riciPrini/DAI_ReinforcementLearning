import matplotlib.pyplot as plt
import numpy as np
filename_learning = 'TLC_naive_dqn.png'
filename_waiting = 'waiting_time.png'
filename_avg_reward = 'avg_reward.png'
filename_epsilon_decay = 'epsilon_decay.png'
def plot_imgs(n_games,y,steps_array,eps_history,avg_wait_times_history,avg_rewards_history):
    plot_learning_curve(steps_array, y, eps_history, filename_learning)
    plot_history_times(n_games,avg_wait_times_history,filename_waiting)
    plot_history_times(n_games,avg_rewards_history,filename_avg_reward)
    plot_history_times(n_games,eps_history,filename_epsilon_decay)


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

def plot_history_times(x,avg_wait,filename):
    print("Saving plot...")
    plt.figure()
    plt.plot(range(x), avg_wait, label='Average Wait Time')
    plt.xlabel('Episodes')
    plt.ylabel('Average Wait Time')
    plt.title('Average Wait Time per Episode')
    plt.legend()
    plt.savefig(filename)
