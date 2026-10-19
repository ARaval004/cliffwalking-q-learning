import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gymnasium.wrappers import RecordVideo

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
VIDEOS_DIR = PROJECT_DIR / "videos"

# Create folders if they do not exist
RESULTS_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)

# Q-learning training function
def train_q_learning(alpha=0.1, gamma=0.9, epsilon_decay=0.995, episodes=3000, noise=0.0):
    env = gym.make("CliffWalking-v1")

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    q_table = np.zeros((num_states, num_actions))

    epsilon = 1.0
    epsilon_min = 0.05
    rewards = []

    rng = np.random.default_rng(42)

    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0

        for step in range(200):

            if rng.random() < epsilon:
                action = int(rng.integers(num_actions))
            else:
                action = int(np.argmax(q_table[state]))

            actual_action = action

            if noise > 0 and rng.random() < noise:
                other_actions = []

                for a in range(num_actions):
                    if a != action:
                        other_actions.append(a)

                actual_action = int(rng.choice(other_actions))

            next_state, reward, terminated, truncated, info = env.step(actual_action)
            old_value = q_table[state, action]

            if terminated or truncated:
                target = reward
            else:
                target = reward + gamma * np.max(q_table[next_state])

            q_table[state, action] = old_value + alpha * (target - old_value)
            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    env.close()
    return q_table, rewards


q_table, rewards = train_q_learning()

moving_average = []

for i in range(len(rewards)):
    start = max(0, i - 49)
    average = np.mean(rewards[start:i + 1])
    moving_average.append(average)

plt.figure(figsize=(10, 5))
plt.plot(rewards, alpha=0.3, label="Episode reward")
plt.plot(moving_average, label="50-episode average")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("CliffWalking Q-Learning Baseline")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "baseline_rewards.png")
plt.show()

print("Average reward (first 100 episodes):", np.mean(rewards[:100]))
print("Average reward (last 100 episodes):", np.mean(rewards[-100:]))
print("Q-table shape:", q_table.shape)


# Section D: Learning rate expreriment

learning_rates = [0.1, 0.5, 0.9]

plt.figure(figsize=(10, 5))

for alpha in learning_rates:
    q_table, rewards = train_q_learning(alpha=alpha)

    moving_average = []

    for i in range(len(rewards)):
        start = max(0, i - 49)
        average = np.mean(rewards[start:i + 1])
        moving_average.append(average)

    plt.plot(moving_average, label=f"Alpha = {alpha}")

    print(f"\nLearning rate: {alpha}")
    print("First 100 average:", np.mean(rewards[:100]))
    print("Last 100 average:", np.mean(rewards[-100:]))

plt.xlabel("Episode")
plt.ylabel("Average reward")
plt.title("Learning Rate Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "learning_rate_comparison.png")
plt.show()


# Section E: Discount factor experiment

discount_factors = [0.5, 0.9, 0.99]

plt.figure(figsize=(10, 5))

for gamma in discount_factors:
    q_table, rewards = train_q_learning(alpha=0.5, gamma=gamma)

    moving_average = []

    for i in range(len(rewards)):
        start = max(0, i - 49)
        average = np.mean(rewards[start:i + 1])
        moving_average.append(average)

    plt.plot(moving_average, label=f"Gamma = {gamma}")

    print(f"\nDiscount factor: {gamma}")
    print("First 100 average:", np.mean(rewards[:100]))
    print("Last 100 average:", np.mean(rewards[-100:]))

plt.xlabel("Episode")
plt.ylabel("Average reward")
plt.title("Discount Factor Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "discount_factor_comparison.png")
plt.show()


# Section F: Epsilon decay experiment

epsilon_decays = [0.99, 0.995, 0.999]

plt.figure(figsize=(10, 5))

for decay in epsilon_decays:
    q_table, rewards = train_q_learning(
        alpha=0.5,
        gamma=0.5,
        epsilon_decay=decay
    )

    moving_average = []

    for i in range(len(rewards)):
        start = max(0, i - 49)
        average = np.mean(rewards[start:i + 1])
        moving_average.append(average)

    plt.plot(moving_average, label=f"Epsilon decay = {decay}")

    print(f"\nEpsilon decay: {decay}")
    print("First 100 average:", np.mean(rewards[:100]))
    print("Last 100 average:", np.mean(rewards[-100:]))

plt.xlabel("Episode")
plt.ylabel("Average reward")
plt.title("Epsilon Decay Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "epsilon_decay_comparison.png")
plt.show()

# Section G: Evaluate the best trained agent

# Train using our selected hyperparameters
best_q_table, training_rewards = train_q_learning(
    alpha=0.5,
    gamma=0.5,
    epsilon_decay=0.995
)

env = gym.make("CliffWalking-v1")

evaluation_rewards = []
evaluation_steps = []
successes = 0
cliff_falls = 0

for episode in range(30):
    state, info = env.reset()
    total_reward = 0
    steps = 0

    for step in range(200):

        action = int(np.argmax(best_q_table[state]))

        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        if reward == -100:
            cliff_falls += 1

        state = next_state

        if terminated:
            successes += 1
            break

        if truncated:
            break

    evaluation_rewards.append(total_reward)
    evaluation_steps.append(steps)

env.close()

print("\nFINAL AGENT EVALUATION")
print("Episodes:", 30)
print("Average reward:", np.mean(evaluation_rewards))
print("Average steps:", np.mean(evaluation_steps))
print("Success rate:", successes / 30 * 100, "%")
print("Total cliff falls:", cliff_falls)


# Section H: Stochastic environment experiment

def evaluate_with_noise(q_table, noise, episodes=100):
    env = gym.make("CliffWalking-v1")
    rng = np.random.default_rng(123)

    rewards = []
    steps_list = []
    successes = 0
    cliff_falls = 0

    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0

        for step in range(200):
            action = int(np.argmax(q_table[state]))
            actual_action = action

            if noise > 0 and rng.random() < noise:
                other_actions = []

                for a in range(env.action_space.n):
                    if a != action:
                        other_actions.append(a)

                actual_action = int(rng.choice(other_actions))

            next_state, reward, terminated, truncated, info = env.step(actual_action)

            total_reward += reward
            steps += 1

            if reward == -100:
                cliff_falls += 1

            state = next_state

            if terminated:
                successes += 1
                break

            if truncated:
                break

        rewards.append(total_reward)
        steps_list.append(steps)

    env.close()

    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps_list)
    success_rate = successes / episodes * 100

    return avg_reward, avg_steps, success_rate, cliff_falls


noise_levels = [0.0, 0.10, 0.25]

trained_rewards = []
original_rewards = []

plt.figure(figsize=(10, 5))

for noise in noise_levels:

    # Train a new agent with this amount of noise
    noisy_q_table, rewards = train_q_learning(
        alpha=0.5,
        gamma=0.5,
        epsilon_decay=0.995,
        noise=noise
    )

    # Plot the training moving average
    moving_average = []

    for i in range(len(rewards)):
        start = max(0, i - 49)
        average = np.mean(rewards[start:i + 1])
        moving_average.append(average)

    plt.plot(moving_average, label=f"Noise = {noise}")

    trained_result = evaluate_with_noise(noisy_q_table, noise)
    original_result = evaluate_with_noise(best_q_table, noise)
    trained_rewards.append(trained_result[0])
    original_rewards.append(original_result[0])

    print(f"\nNOISE LEVEL: {noise}")
    print("Agent trained at this noise level:")
    print("Average reward:", trained_result[0])
    print("Average steps:", trained_result[1])
    print("Success rate:", trained_result[2], "%")
    print("Cliff falls:", trained_result[3])

    print("Original agent under this noise:")
    print("Average reward:", original_result[0])
    print("Average steps:", original_result[1])
    print("Success rate:", original_result[2], "%")
    print("Cliff falls:", original_result[3])


plt.xlabel("Episode")
plt.ylabel("Average reward")
plt.title("Training with Different Noise Levels")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "noise_training_comparison.png")
plt.show()


plt.figure(figsize=(8, 5))

plt.plot(noise_levels, trained_rewards, marker="o",
         label="Trained with noise")

plt.plot(noise_levels, original_rewards, marker="o",
         label="Original agent")

plt.xlabel("Noise probability")
plt.ylabel("Average evaluation reward")
plt.title("Agent Performance Under Noise")
plt.xticks(noise_levels)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "noise_evaluation_comparison.png")
plt.show()

video_env = gym.make("CliffWalking-v1", render_mode="rgb_array")

video_env = RecordVideo(
    video_env,
    video_folder=str(VIDEOS_DIR),
    episode_trigger=lambda episode: episode == 0,
    name_prefix="cliffwalking_final",
    fps=3
)


state, info = video_env.reset()
total_reward = 0
steps = 0
reached_goal = False

for step in range(200):
    action = int(np.argmax(best_q_table[state]))
    state, reward, terminated, truncated, info = video_env.step(action)
    total_reward += reward
    steps += 1

    if terminated:
        reached_goal = True
        break

    if truncated:
        break

video_env.close()

print("\nVIDEO RESULTS")
print("Reached goal:", reached_goal)
print("Total steps:", steps)
print("Total reward:", total_reward)
print("Video saved in the videos folder")
