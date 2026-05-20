from collections import defaultdict
import gymnasium as gym
import numpy as np

class BlackjackTest:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 1.0,
            learning_rate_decay: float = 1.0,
            final_learning_rate: float | None = None,
        ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.lr_decay = learning_rate_decay
        self.final_lr = final_learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
    
    def get_action(self, obs: tuple[int, int, bool]) -> int:
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        # Q_learning kernel function
        if terminated:
            future_q_value = 0.0
        else:
            future_q_value = np.max(self.q_values[next_obs])

        target = reward + self.discount_factor * future_q_value

        temporal_difference = target - self.q_values[obs][action]   # TD error

        self.q_values[obs][action] += self.lr * temporal_difference

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def decay_learning_rate(self):
        self.lr = max(self.final_lr or 0.0, self.lr * self.lr_decay)


learning_rate = 0.05
n_episodes = 200_000
start_epsilon = 1.0
final_epsilon = 0.001
epsilon_decay = 0.99993
lr_decay = 0.99998
final_lr = 0.001


env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackTest(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    learning_rate_decay=lr_decay,
    final_learning_rate=final_lr,
)

from tqdm import tqdm

for episode in tqdm(range(n_episodes)):
    obs, _ = env.reset()
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, done, truncated, _ = env.step(action)

        agent.update(obs, action, reward, done, next_obs)
        done = done or truncated
        obs = next_obs

    agent.decay_epsilon()
    agent.decay_learning_rate()


# 训练结果可视化
from matplotlib import pyplot as plt

def get_moving_avg(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

axs[0].set_title("rewards")
reward_moving_avg = get_moving_avg(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_avg)), reward_moving_avg)
axs[0].set_ylabel("avg_reward")
axs[0].set_xlabel("episodes")

axs[1].set_title("episode lengths")
length_moving_avg = get_moving_avg(
    env.length_queue,
    rolling_length,
    "valid"
) 
axs[1].plot(range(len(length_moving_avg)), length_moving_avg)
axs[1].set_ylabel("avg_length")
axs[1].set_xlabel("episodes")

axs[2].set_title("training error")
training_error_moving_avg = get_moving_avg(
    agent.training_error,
    rolling_length,
    "valid"
)
axs[2].plot(range(len(training_error_moving_avg)), training_error_moving_avg)
axs[2].set_ylabel("avg_training_error")
axs[2].set_xlabel("episodes")

plt.tight_layout()
plt.savefig("blackjack_training_results.png", dpi=300, bbox_inches='tight')


def test_agent(agent: BlackjackTest, n_episodes: int=1000):
    total_rewards = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            done = done or truncated
            # obs = next_obs
        total_rewards.append(episode_reward)
    agent.epsilon = old_epsilon
    win_rate = np.mean(np.array(total_rewards) > 0)
    avg_reward = np.mean(total_rewards)

    print(f"Win rate over {n_episodes} episodes: {win_rate:.2%}")
    print(f"Average reward over {n_episodes} episodes: {avg_reward}")

test_agent(agent)
