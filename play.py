from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm


class BlackjackAgent:
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

    def update(self, obs, action, reward, terminated, next_obs):
        future_q_value = 0.0 if terminated else np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q_value
        td_error = target - self.q_values[obs][action]
        self.q_values[obs][action] += self.lr * td_error
        self.training_error.append(td_error)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def decay_learning_rate(self):
        self.lr = max(self.final_lr or 0.0, self.lr * self.lr_decay)


def train_agent():
    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=200_000)

    agent = BlackjackAgent(
        env=env,
        learning_rate=0.05,
        initial_epsilon=1.0,
        epsilon_decay=0.99993,
        final_epsilon=0.001,
        learning_rate_decay=0.99998,
        final_learning_rate=0.001,
    )

    print("Training AI agent (200,000 episodes)...")
    for episode in tqdm(range(200_000)):
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

    return agent


CARD_NAMES = {1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7",
              8: "8", 9: "9", 10: "10"}
ACTION_NAMES = {0: "STICK(停牌)", 1: "HIT(要牌)"}


def card_name(value: int) -> str:
    return CARD_NAMES.get(value, str(value))


def display_hand(hand_num: int, obs: tuple, ai_action: int):
    player_sum, dealer_card, usable_ace = obs
    print()
    print("=" * 44)
    print(f"  Hand #{hand_num}")
    print(f"  Your hand sum:     {player_sum}")
    print(f"  Dealer shows:      {card_name(dealer_card)}")
    print(f"  Usable Ace:        {'Yes' if usable_ace else 'No'}")
    print()
    print(f"  [ AI recommends: {ACTION_NAMES[ai_action]} ]")
    print("=" * 44)


def display_result(reward: float):
    if reward > 0:
        print("\n  >>> You WIN! <<<")
    elif reward == 0:
        print("\n  >>> PUSH (Draw) <<<")
    else:
        print("\n  >>> You LOSE <<<")


def display_stats(stats: dict):
    total = stats["wins"] + stats["losses"] + stats["draws"]
    if total == 0:
        return
    print()
    print("-" * 44)
    print(f"  Session: {stats['wins']}W / {stats['losses']}L / {stats['draws']}D"
          f"  ({stats['wins'] / total * 100:.1f}% win rate)")
    if stats["agreed"] > 0:
        print(f"  Followed AI advice: {stats['agreed']}/{total} times"
              f"  ({stats['agreed_wins'] / stats['agreed'] * 100:.1f}% won)")
    print("-" * 44)


def play_game(agent: BlackjackAgent):
    env = gym.make("Blackjack-v1", sab=False)
    agent.epsilon = 0.0

    stats = {"wins": 0, "losses": 0, "draws": 0, "agreed": 0, "agreed_wins": 0}
    hand_num = 0

    print("\n" + "=" * 44)
    print("  Welcome to Blackjack!")
    print("  The AI advisor has been trained.")
    print("  h = hit(要牌)  s = stick(停牌)  q = quit(退出)")
    print("=" * 44)

    while True:
        hand_num += 1
        obs, _ = env.reset()
        done = False

        while not done:
            ai_action = agent.get_action(obs)
            display_hand(hand_num, obs, ai_action)

            while True:
                choice = input("  (h)it=要牌 / (s)tick=停牌 / (q)uit=退出: ").strip().lower()
                if choice in ("h", "hit"):
                    user_action = 1
                    break
                elif choice in ("s", "stick"):
                    user_action = 0
                    break
                elif choice in ("q", "quit"):
                    print("\n  Thanks for playing!")
                    display_stats(stats)
                    return
                else:
                    print("  Please enter h, s, or q.")

            followed_ai = (user_action == ai_action)
            if followed_ai:
                stats["agreed"] += 1

            obs, reward, done, truncated, _ = env.step(user_action)
            done = done or truncated

            if done:
                display_result(reward)
                if reward > 0:
                    stats["wins"] += 1
                    if followed_ai:
                        stats["agreed_wins"] += 1
                elif reward == 0:
                    stats["draws"] += 1
                else:
                    stats["losses"] += 1

        display_stats(stats)

        while True:
            again = input("\n  Play again? (y/n): ").strip().lower()
            if again in ("y", "yes", ""):
                break
            elif again in ("n", "no"):
                print("\n  Thanks for playing!")
                display_stats(stats)
                return
            else:
                print("  Please enter y or n.")


if __name__ == "__main__":
    agent = train_agent()
    play_game(agent)
