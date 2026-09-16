"""
Gridworld Reinforcement Learning Example (Policy Gradient / REINFORCE)

This example demonstrates how to train a neural network policy to navigate a
4x4 Gridworld environment around obstacles towards a goal using Reinforcement
Learning with `train_with_advantages`.

How Reinforcement Learning works in this library:
- In supervised training (`train`), the caller supplies fixed ground-truth targets.
- In reinforcement learning (`train_with_advantages`), the network generates its own
  actions by sampling from its output distribution (the policy).
- During each episode, the agent collects trajectory samples (states, actions, rewards).
  At the end of the episode, discounted returns and advantages are computed:
    * Reaching Goal (G): Positive advantage scaled by path efficiency, reinforcing
      the sequence of actions that reached the target in the fewest steps.
    * Timeout / Wandering: Negative advantage, penalising trajectories that failed
      to reach the goal.
- `train_with_advantages(states, action_targets, advantages)` applies the policy gradient
  update: it scales the output delta (prediction - chosen_action) by the advantage before
  backpropagating through all hidden layers, shifting the network towards optimal paths.

Environment:
- 4x4 grid where:
    S = Start position (0, 0)
    . = Open walkable cell
    X = Obstacle / Wall
    G = Goal position (3, 3)
- 4 discrete actions: UP (0), DOWN (1), LEFT (2), RIGHT (3)
- Shortest geometric path around obstacles is exactly 6 steps.

Workflow:
1. Demonstrate baseline navigation with an untrained agent (random policy).
2. Build and train a 16-input, 32-hidden, 4-output Softmax policy network.
3. Evaluate the trained policy and visualise the optimal 6-step path.
4. Render the full learned navigation policy map across all grid cells.
"""

import os
import sys
import random

# Add the directory containing the compiled .pyd module to the import search path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PYTHON_DIR)
sys.path.append(os.path.join(PYTHON_DIR, "x64", "Release"))

try:
    import neuralnetwork as nn
    print("Successfully imported neuralnetwork library!")
except ImportError as e:
    print(f"Error: Could not import neuralnetwork extension module: {e}")
    print("Please make sure you have built the solution in Release/x64 first.")
    sys.exit(1)


# 1. Setup the 4x4 Gridworld environment
# S = Start, . = Open space, X = Obstacle / Wall, G = Goal
GRID = [
    ["S", ".", ".", "."],
    [".", "X", "X", "."],
    [".", ".", "X", "."],
    [".", ".", ".", "G"]
]
ROWS = 4
COLS = 4

ACTIONS = [0, 1, 2, 3]  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ACTION_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_SYMBOLS = {0: "  ^  ", 1: "  v  ", 2: "  <  ", 3: "  >  "}

# One-hot vector representing the agent's grid index (16 coordinates)
INPUT_SIZE = ROWS * COLS
OUTPUT_SIZE = len(ACTIONS)


def get_state_vector(r, c):
    """Converts a row and column location into a 1D state vector for the network."""
    state = [0.0] * INPUT_SIZE
    state[r * COLS + c] = 1.0
    return state


def get_valid_actions(r, c):
    """Returns a list of action indices that do not cross grid boundaries or hit obstacles."""
    valid_actions = []
    for action, (dr, dc) in ACTION_MOVES.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and GRID[nr][nc] != "X":
            valid_actions.append(action)
    return valid_actions


def get_next_state_and_reward(r, c, action):
    """Executes a movement action and returns next position, reward, and terminal flag."""
    dr, dc = ACTION_MOVES[action]
    nr, nc = r + dr, c + dc

    # Boundary or obstacle check
    if not (0 <= nr < ROWS and 0 <= nc < COLS) or GRID[nr][nc] == "X":
        return r, c, -0.2, False  # Stays in place, small penalty

    if GRID[nr][nc] == "G":
        return nr, nc, 1.0, True   # Goal reached

    return nr, nc, -0.02, False   # Small step cost to encourage shortest paths


def format_grid(path_history=None):
    """Renders a formatted ASCII grid showing obstacles, start, goal, and step markers."""
    step_indices = {}
    if path_history:
        for step_idx, (r, c) in enumerate(path_history):
            if (r, c) not in step_indices:
                step_indices[(r, c)] = step_idx

    divider = "+-----" * COLS + "+"
    lines = [divider]

    for r in range(ROWS):
        row_cells = []
        for c in range(COLS):
            cell_type = GRID[r][c]
            if (r, c) in step_indices:
                step_num = step_indices[(r, c)]
                if cell_type == "G":
                    cell_str = " [G] "
                elif cell_type == "S" and step_num == 0:
                    cell_str = " [S] "
                else:
                    cell_str = f"[{step_num:2d}] "
            elif cell_type == "X":
                cell_str = "  X  "
            elif cell_type == "G":
                cell_str = "  G  "
            elif cell_type == "S":
                cell_str = "  S  "
            else:
                cell_str = "  .  "
            row_cells.append(cell_str)
        lines.append("|" + "|".join(row_cells) + "|")
        lines.append(divider)

    return "\n".join(lines)


def format_policy_map(net):
    """Visualises the optimal greedy action direction for every accessible grid cell."""
    divider = "+-----" * COLS + "+"
    lines = [divider]

    for r in range(ROWS):
        row_cells = []
        for c in range(COLS):
            if GRID[r][c] == "X":
                row_cells.append("  X  ")
            elif GRID[r][c] == "G":
                row_cells.append("  G  ")
            else:
                best_action = select_greedy_action(net, r, c)
                row_cells.append(ACTION_SYMBOLS.get(best_action, "  ?  "))
        lines.append("|" + "|".join(row_cells) + "|")
        lines.append(divider)

    return "\n".join(lines)


def select_action(net, r, c, epsilon=0.0):
    """
    Selects an action using epsilon-greedy exploration and masked policy distribution.
    Returns: (chosen_action, one_hot_action_target)
    """
    valid_actions = get_valid_actions(r, c)
    if not valid_actions:
        return 0, [0.0] * OUTPUT_SIZE

    if epsilon > 0.0 and random.random() < epsilon:
        action = random.choice(valid_actions)
    else:
        state_v = get_state_vector(r, c)
        probabilities = net.think(state_v)

        # Mask invalid actions
        masked_probabilities = [probabilities[a] if a in valid_actions else 0.0 for a in ACTIONS]
        total_probability = sum(masked_probabilities)

        if total_probability > 0.0:
            normalised = [p / total_probability for p in masked_probabilities]
            roll = random.random()
            cumulative = 0.0
            action = valid_actions[-1]
            for a in valid_actions:
                cumulative += normalised[a]
                if roll <= cumulative:
                    action = a
                    break
        else:
            action = random.choice(valid_actions)

    action_target = [0.0] * OUTPUT_SIZE
    action_target[action] = 1.0
    return action, action_target


def select_greedy_action(net, r, c):
    """Selects the highest-probability valid action deterministically for inference."""
    valid_actions = get_valid_actions(r, c)
    if not valid_actions:
        return 0
    state_v = get_state_vector(r, c)
    probabilities = net.think(state_v)
    return max(valid_actions, key=lambda a: probabilities[a])


def build_policy_network(learning_rate=0.005):
    """
    Constructs a Feed-Forward policy neural network:
    - Input: 16 units (one-hot agent position in 4x4 grid)
    - Hidden: 32 units with ReLU activation and Adam optimiser
    - Output: 4 units with Softmax activation and CrossEntropy loss
    """
    topology = [INPUT_SIZE, 32, OUTPUT_SIZE]

    hidden_activation = nn.Activation(nn.ActivationMethod.Relu, 0.0)
    hidden_layers = [
        nn.LayerDetails(
            nn.LayerArchitecture.FF,
            32,
            hidden_activation,
            0.0,                    # Dropout rate
            0.001,                  # Weight decay
            nn.OptimiserType.Adam,  # Adam optimiser
            0.9                     # Momentum factor
        )
    ]

    output_activation = nn.Activation(nn.ActivationMethod.Softmax, 0.0, 1.0)
    output_layer = nn.OutputLayerDetails(
        OUTPUT_SIZE,
        output_activation,
        nn.ErrorCalculationType.CrossEntropy,
        nn.EvaluationConfig(),
        0.001,                  # Weight decay
        nn.OptimiserType.Adam,
        0.9                     # Momentum factor
    )

    options = (
        nn.NeuralNetworkOptions.create(topology)
        .with_hidden_layers(hidden_layers)
        .with_output_layer_details(output_layer)
        .with_learning_rate(learning_rate)
        .with_batch_size(16)
        .with_number_of_epoch(1)
        .with_has_bias(True)
        .with_clip_threshold(0.5)
        .with_enable_bptt(False)
        .with_log_level(nn.LogLevel.Warning)
        .build()
    )

    return nn.NeuralNetwork(options)


def run_episode(net=None, epsilon=0.0, max_steps=25, greedy=False):
    """
    Simulates an entire traversal timeline from start to goal or step limit.
    Returns: (path_history, actions_taken, reached_goal)
    """
    r, c = 0, 0
    path = [(r, c)]
    actions_taken = []
    reached_goal = False

    for _ in range(max_steps):
        if GRID[r][c] == "G":
            reached_goal = True
            break

        if net is None:
            valid_actions = get_valid_actions(r, c)
            action = random.choice(valid_actions)
        elif greedy:
            action = select_greedy_action(net, r, c)
        else:
            action, _ = select_action(net, r, c, epsilon=epsilon)

        next_r, next_c, _, goal = get_next_state_and_reward(r, c, action)
        actions_taken.append(action)
        r, c = next_r, next_c
        path.append((r, c))

        if goal:
            reached_goal = True
            break

    return path, actions_taken, reached_goal


def train_agent(net, total_episodes=150, discount_factor=0.95):
    """
    Trains the policy network using Reinforcement Learning with advantage scaling.
    """
    print(f"\n--- Training the Policy Network ({total_episodes} episodes) ---")

    batch_states = []
    batch_actions = []
    batch_advantages = []

    successes = 0

    for episode in range(1, total_episodes + 1):
        # Epsilon exploration decays from 40% down to 5%
        epsilon = max(0.05, 0.40 * (1.0 - episode / total_episodes))

        r, c = 0, 0
        episode_states = []
        episode_actions = []
        episode_rewards = []
        reached_goal = False

        for _ in range(30):
            state_v = get_state_vector(r, c)
            action, action_target = select_action(net, r, c, epsilon=epsilon)
            next_r, next_c, reward, reached_goal = get_next_state_and_reward(r, c, action)

            episode_states.append(state_v)
            episode_actions.append(action_target)
            episode_rewards.append(reward)

            r, c = next_r, next_c

            if reached_goal:
                break

        if reached_goal:
            successes += 1

        # Calculate discounted advantages
        T = len(episode_rewards)
        if reached_goal:
            # Reward shorter paths with higher advantage bonus
            path_efficiency = (30 - T) / 30.0
            episode_advantages = [
                0.2 + 0.8 * (discount_factor ** (T - 1 - t)) * path_efficiency
                for t in range(T)
            ]
        else:
            # Penalise wandering or timing out
            episode_advantages = [-0.2 for _ in range(T)]

        batch_states.extend(episode_states)
        batch_actions.extend(episode_actions)
        batch_advantages.extend(episode_advantages)

        # Batch update policy network
        if len(batch_states) >= 16 or episode == total_episodes:
            if batch_states:
                net.train_with_advantages(batch_states, batch_actions, batch_advantages)
                batch_states.clear()
                batch_actions.clear()
                batch_advantages.clear()

        # Log training statistics
        if episode % 30 == 0:
            success_rate = (successes / 30.0) * 100.0
            print(f"Episode {episode:3d}/{total_episodes:3d} | Recent Goal Reach Rate: {success_rate:5.1f}%")
            successes = 0

    print("--- Training Completed Successfully ---\n")


def print_step_by_step(path, actions):
    """Prints a clear step-by-step description of each transition along the path."""
    for idx, action in enumerate(actions):
        from_r, from_c = path[idx]
        to_r, to_c = path[idx + 1]
        name = ACTION_NAMES[action]
        goal_flag = " [GOAL REACHED]" if GRID[to_r][to_c] == "G" else ""
        print(f"  Step {idx + 1:2d}: ({from_r}, {from_c}) -> Action: {name:5s} -> ({to_r}, {to_c}){goal_flag}")


def main():
    random.seed(42)

    # Number of training episodes (defaults to 150; can be overridden via env var)
    total_episodes = int(os.environ.get("GRIDWORLD_EPISODES", "150"))

    # =====================================================================
    # INTERFACE 1: SYSTEM PRE-TRAINING VISUALISATION
    # =====================================================================
    print("=" * 65)
    print(" 1. BEFORE TRAINING (Pure Random Baseline)")
    print("=" * 65)
    print("Environment Layout (S = Start, G = Goal, X = Obstacle):\n")
    print(format_grid())
    print("\nSimulating navigation with an untrained random policy:")
    untrained_path, untrained_actions, untrained_goal = run_episode(net=None, max_steps=20)
    print(format_grid(untrained_path))
    steps_taken = len(untrained_actions)
    status = "Reached Goal" if untrained_goal else "Timed out without reaching Goal"
    print(f"Outcome: {status} (Steps taken: {steps_taken})\n")

    # =====================================================================
    # TRAINING THE POLICY NETWORK VIA REINFORCEMENT LEARNING
    # =====================================================================
    net = build_policy_network(learning_rate=0.005)
    train_agent(net, total_episodes=total_episodes, discount_factor=0.95)

    # =====================================================================
    # INTERFACE 2: SYSTEM POST-TRAINING OPTIMAL TRAJECTORY
    # =====================================================================
    print("=" * 65)
    print(" 2. AFTER TRAINING (Optimised Neural Policy)")
    print("=" * 65)
    print("The trained network navigates cleanly around obstacles straight to Goal:\n")
    trained_path, trained_actions, trained_goal = run_episode(net=net, greedy=True, max_steps=20)
    print(format_grid(trained_path))
    print(f"Total actions taken: {len(trained_actions)} (Optimal geometric path: 6 steps)\n")

    print("Step-by-step navigation log:")
    print_step_by_step(trained_path, trained_actions)

    # =====================================================================
    # INTERFACE 3: LEARNED POLICY DIRECTION MAP
    # =====================================================================
    print("\n" + "=" * 65)
    print(" 3. LEARNED POLICY MAP (Preferred Direction for Every Cell)")
    print("=" * 65)
    print("Direction indicators: ^ (UP), v (DOWN), < (LEFT), > (RIGHT)\n")
    print(format_policy_map(net))
    print()


if __name__ == "__main__":
    main()
