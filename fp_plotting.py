import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, num_episodes, loss_function_returns, episode_rewards, episode_reward_running_avgs):
        self.eps = num_episodes
        self.losses = loss_function_returns
        self.rewards = episode_rewards
        self.reward_avgs = episode_reward_running_avgs


    def plot(self):
        plt.title("Loss function return value over time")
        plt.xlabel("# call to loss function")
        plt.ylabel("Loss function return value (log scale)")
        plt.semilogy(range(1, len(self.losses) + 1), self.losses, linewidth=2.0, c="red")
        plt.show()

        plt.title("Raw reward per episode over episode iterations")
        plt.xlabel("Episode #")
        plt.ylabel("Cumulative reward")
        plt.plot(range(1, self.eps+1), self.rewards, linewidth=2.0, c="green")
        plt.show()


        plt.title("Average reward over episode iterations")
        plt.xlabel("Episode #")
        plt.ylabel("Average reward of current episode and past episodes")
        plt.plot(range(1, self.eps+1), self.reward_avgs, linewidth=2.0, c="blue")
        plt.show()


        with open("./training_results.py", 'w') as f:
            f.write(str(self.rewards) + "\n")
            f.write(str(self.reward_avgs) + "\n")
