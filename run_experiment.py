import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import CleaningEnv
from agent import QLearningAgent

EPISODES = 6000
SEEDS = 10
WINDOW = 100       


def train(reward_mode, seed):
    env = CleaningEnv(reward_mode)
    agent = QLearningAgent(env.n_states, env.n_actions, seed=seed)
    success = np.zeros(EPISODES)
    for ep in range(EPISODES):
        epsilon = max(0.05, 1.0 - ep / (0.6 * EPISODES))   
        state, done = env.reset(), False
        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done, cleaned = env.step(action)
            agent.update(state, action, reward, next_state, cleaned)
            state = next_state
        success[ep] = cleaned
    return agent, success


def evaluate(agent, reward_mode, show=False):
    env = CleaningEnv(reward_mode)
    state, done = env.reset(), False
    while not done:
        if show:
            env.render()
        state, _, done, cleaned = env.step(agent.act(state, epsilon=0.0))
    return cleaned, env.steps


def random_baseline(episodes=500):
    env, rng, wins = CleaningEnv(), np.random.default_rng(0), 0
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            _, _, done, cleaned = env.step(int(rng.integers(4)))
        wins += cleaned
    return wins / episodes


if __name__ == "__main__":
    print(f"Random policy success rate: {random_baseline():.0%}\n")
    plt.figure(figsize=(7, 4))
    for mode in ["dense", "sparse"]:
        curves, results, reach90 = [], [], []
        for seed in range(SEEDS):
            agent, success = train(mode, seed)
            curve = success.reshape(-1, WINDOW).mean(axis=1)
            curves.append(curve)
            results.append(evaluate(agent, mode))
            hit = np.where(curve >= 0.9)[0]
            reach90.append((hit[0] + 1) * WINDOW if len(hit) else None)
        curves = np.array(curves)
        solved = [r for r in results if r[0]]
        reached = [x for x in reach90 if x is not None]
        print(f"{mode:>6}: learned policy cleans the room in {len(solved)}/{SEEDS} seeds, "
              f"mean steps {np.mean([s for _, s in solved]) if solved else float('nan'):.1f}, "
              f"episodes to 90% success: {np.mean(reached) if reached else float('nan'):.0f} "
              f"({len(reached)}/{SEEDS} seeds reached it)")
        x = (np.arange(curves.shape[1]) + 1) * WINDOW
        mean, std = curves.mean(axis=0), curves.std(axis=0)
        plt.plot(x, mean, label=f"{mode} reward")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Training episode")
    plt.ylabel("Success rate")
    plt.title(f"Q-learning cleaning robot, mean of {SEEDS} seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150)

    #Watch the robot clean the room
    agent, _ = train("sparse", seed=0)
    evaluate(agent, "sparse", show=True)