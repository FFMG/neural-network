"""
Tic-Tac-Toe Reinforcement Learning Example (Policy Gradient / REINFORCE)

This example demonstrates how to train a neural network policy to play
Tic-Tac-Toe using Reinforcement Learning with `train_with_advantages`.

How Reinforcement Learning works in this library:
- In supervised training (`train`), the caller supplies fixed ground-truth targets.
- In reinforcement learning (`train_with_advantages`), the network generates its own
  actions by sampling from its output distribution (the policy). At the end of each game,
  the environment awards an outcome-dependent scalar advantage:
    * Win:  +1.0 (reinforces the actions that led to victory)
    * Draw: +0.2 (positive reward for securing a draw rather than losing)
    * Loss: -1.0 (penalises the actions that led to defeat)
- `train_with_advantages(states, action_targets, advantages)` applies the policy gradient
  update: it scales the output delta (prediction - chosen_action) by the advantage before
  backpropagating through all hidden layers, shifting the network towards winning strategies.

Workflow:
1. Configure a 9-input, 36-hidden, 9-output Softmax policy network.
2. Train the policy over a series of games where advantages are computed from game outcomes.
3. Evaluate the trained policy over 100 games against a Random opponent.
4. Demonstrate a single move-by-move match with board visualisation.
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


# Winning cell combinations on a 3x3 board (indices 0 to 8)
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Horizontal rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Vertical columns
    (0, 4, 8), (2, 4, 6)              # Diagonals
]


def check_winner(board):
    """
    Check if either player has won the game.
    Returns:
      +1.0 if Player X (Neural Network) has won
      -1.0 if Player O (Opponent) has won
       0.0 if the game is still undecided or drawn
    """
    for a, b, c in WIN_LINES:
        total = board[a] + board[b] + board[c]
        if total == 3.0:
            return 1.0   # Player X wins
        if total == -3.0:
            return -1.0  # Player O wins
    return 0.0


def is_board_full(board):
    """Returns True if no empty cells remain on the board."""
    return all(cell != 0.0 for cell in board)


def get_available_moves(board):
    """Returns a list of cell indices (0..8) that are currently empty."""
    return [i for i, cell in enumerate(board) if cell == 0.0]


def format_board(board):
    """Returns a user-friendly 3x3 text visualisation of the board."""
    symbols = {1.0: "X", -1.0: "O", 0.0: "."}
    lines = []
    for r in range(3):
        row = [symbols[board[r * 3 + c]] for c in range(3)]
        lines.append(" " + " | ".join(row))
        if r < 2:
            lines.append("---+---+---")
    return "\n".join(lines)


def select_action(net, board, epsilon=0.0):
    """
    Select an action for Player X using the neural network's policy.
    - Uses epsilon-greedy exploration during training.
    - Masks invalid moves (already occupied cells) so illegal moves are never picked.
    - Returns: (chosen_cell_index, one_hot_action_vector)
    """
    valid_moves = get_available_moves(board)
    if not valid_moves:
        return None, None

    # Epsilon exploration: pick a random valid move
    if epsilon > 0.0 and random.random() < epsilon:
        action = random.choice(valid_moves)
    else:
        # Evaluate model probabilities for all 9 cells
        probabilities = net.think(board)

        # Mask invalid (already occupied) cells with zero probability
        masked_probs = [probabilities[i] if i in valid_moves else 0.0 for i in range(9)]
        total_prob = sum(masked_probs)

        if total_prob > 0.0:
            # Normalise over valid moves and sample from distribution
            normalised = [p / total_prob for p in masked_probs]
            r = random.random()
            cumulative = 0.0
            action = valid_moves[-1]
            for move in valid_moves:
                cumulative += normalised[move]
                if r <= cumulative:
                    action = move
                    break
        else:
            action = random.choice(valid_moves)

    # Build a one-hot action target vector of length 9
    action_target = [0.0] * 9
    action_target[action] = 1.0
    return action, action_target


def play_random_move(board):
    """Opponent move: selects an available cell uniformly at random."""
    valid_moves = get_available_moves(board)
    if not valid_moves:
        return None
    return random.choice(valid_moves)


def build_policy_network(learning_rate=0.01):
    """
    Builds a Feed-Forward policy network:
    - Input layer: 9 neurons (representing 9 board cells: 0.0=empty, +1.0=X, -1.0=O)
    - Hidden layer: 36 neurons with ReLU activation and Adam optimiser
    - Output layer: 9 neurons with Softmax activation and CrossEntropy loss
    """
    topology = [9, 36, 9]

    hidden_activation = nn.Activation(nn.ActivationMethod.Relu, 0.0)
    hidden_layers = [
        nn.LayerDetails(
            nn.LayerArchitecture.FF,
            36,
            hidden_activation,
            0.0,                     # No dropout
            0.0,                     # No weight decay
            nn.OptimiserType.Adam,   # Adam optimiser
            0.9                      # Momentum factor
        )
    ]

    output_activation = nn.Activation(nn.ActivationMethod.Softmax, 0.0, 1.0)
    output_layer = nn.OutputLayerDetails(
        9,
        output_activation,
        nn.ErrorCalculationType.CrossEntropy,
        nn.EvaluationConfig(),
        0.0,
        nn.OptimiserType.Adam,
        0.9
    )

    options = (
        nn.NeuralNetworkOptions.create(topology)
        .with_hidden_layers(hidden_layers)
        .with_output_layer_details(output_layer)
        .with_learning_rate(learning_rate)
        .with_batch_size(16)
        .with_number_of_epoch(1)
        .with_has_bias(True)
        .with_enable_bptt(False)
        .with_log_level(nn.LogLevel.Warning)
        .build()
    )

    return nn.NeuralNetwork(options)


def train_agent(net, total_episodes=1500, discount_factor=0.95):
    """
    Trains the neural network using Reinforcement Learning against a simulated opponent.
    - Wins receive +1.0 reward
    - Draws receive +0.2 reward
    - Losses receive -1.0 penalty
    """
    print(f"\n--- Starting Reinforcement Learning Training ({total_episodes} episodes) ---")

    # Buffer for batched updates
    batch_states = []
    batch_actions = []
    batch_advantages = []

    wins = 0
    draws = 0
    losses = 0

    for episode in range(1, total_episodes + 1):
        # Initialise empty board (9 cells: 0.0 = empty)
        board = [0.0] * 9
        episode_states = []
        episode_actions = []

        # Decay exploration rate over training (from 30% down to 5%)
        epsilon = max(0.05, 0.30 * (1.0 - episode / total_episodes))

        # Play one game
        game_over = False
        winner = 0.0

        while not game_over:
            # 1. Neural Network move (Player X)
            action, action_target = select_action(net, board, epsilon=epsilon)
            if action is None:
                break

            # Record state and action taken by Player X
            episode_states.append(list(board))
            episode_actions.append(action_target)

            board[action] = 1.0  # Apply X's move

            winner = check_winner(board)
            if winner != 0.0 or is_board_full(board):
                game_over = True
                break

            # 2. Opponent move (Player O)
            opp_action = play_random_move(board)
            if opp_action is not None:
                board[opp_action] = -1.0  # Apply O's move

            winner = check_winner(board)
            if winner != 0.0 or is_board_full(board):
                game_over = True
                break

        # Assign rewards based on game outcome
        if winner == 1.0:
            reward = 1.0
            wins += 1
        elif winner == -1.0:
            reward = -1.0
            losses += 1
        else:
            reward = 0.2  # Positive incentive for securing a draw over losing
            draws += 1

        # Calculate discounted advantage for each step in this episode:
        # Steps closer to the terminal state receive a slightly higher weight.
        T = len(episode_states)
        for t in range(T):
            advantage = reward * (discount_factor ** (T - 1 - t))
            batch_states.append(episode_states[t])
            batch_actions.append(episode_actions[t])
            batch_advantages.append(advantage)

        # Flush batched updates every 32 samples or at end of episodes
        if len(batch_states) >= 32 or episode == total_episodes:
            if batch_states:
                net.train_with_advantages(batch_states, batch_actions, batch_advantages)
                batch_states.clear()
                batch_actions.clear()
                batch_advantages.clear()

        # Log training statistics every 300 games
        if episode % 300 == 0:
            rate = (wins / episode) * 100.0
            print(f"Episode {episode:4d}/{total_episodes} | Wins: {wins:4d} | Draws: {draws:4d} | Losses: {losses:4d} | Win Rate: {rate:5.1f}%")

    final_win_rate = (wins / total_episodes) * 100.0
    print(f"--- Training Completed: Final Win Rate = {final_win_rate:.1f}% ---\n")


def evaluate_against_random(net, num_games=100):
    """
    Evaluates the trained model against a pure Random player over a series of matches.
    In evaluation mode, the policy acts greedily without random exploration (epsilon = 0.0).
    """
    print(f"--- Evaluating Trained Model Against Random Player ({num_games} games) ---")
    wins = 0
    draws = 0
    losses = 0

    for _ in range(num_games):
        board = [0.0] * 9
        game_over = False

        while not game_over:
            # Trained model move (X)
            action, _ = select_action(net, board, epsilon=0.0)
            if action is None:
                break
            board[action] = 1.0

            winner = check_winner(board)
            if winner != 0.0 or is_board_full(board):
                break

            # Random opponent move (O)
            opp_action = play_random_move(board)
            if opp_action is not None:
                board[opp_action] = -1.0

            winner = check_winner(board)
            if winner != 0.0 or is_board_full(board):
                break

        if winner == 1.0:
            wins += 1
        elif winner == -1.0:
            losses += 1
        else:
            draws += 1

    win_rate = (wins / num_games) * 100.0
    draw_rate = (draws / num_games) * 100.0
    loss_rate = (losses / num_games) * 100.0

    print(f"Results over {num_games} games:")
    print(f"  * Wins:   {wins:3d} ({win_rate:5.1f}%)")
    print(f"  * Draws:  {draws:3d} ({draw_rate:5.1f}%)")
    print(f"  * Losses: {losses:3d} ({loss_rate:5.1f}%)")
    print(f"  * Non-loss Rate (Wins + Draws): {win_rate + draw_rate:5.1f}%\n")


def play_demonstration_game(net):
    """
    Plays a single demonstration game against a Random opponent,
    printing the board step-by-step so the reader can follow the game.
    """
    print("====================================================")
    print("      Demonstration Match: Trained Model vs Random   ")
    print("         Model = 'X'   |   Random Player = 'O'      ")
    print("====================================================")

    board = [0.0] * 9
    turn = 1

    print("\nInitial empty board:")
    print(format_board(board))

    while True:
        # Player X (Trained Model)
        action, _ = select_action(net, board, epsilon=0.0)
        if action is None:
            break
        board[action] = 1.0
        print(f"\n--- Turn {turn}: Trained Model (X) plays cell {action} ---")
        print(format_board(board))

        winner = check_winner(board)
        if winner == 1.0:
            print("\nOutcome: Trained Model (X) won the game!")
            break
        if is_board_full(board):
            print("\nOutcome: Game ended in a Draw!")
            break

        # Player O (Random Player)
        opp_action = play_random_move(board)
        if opp_action is None:
            break
        board[opp_action] = -1.0
        print(f"\n--- Turn {turn}: Random Player (O) plays cell {opp_action} ---")
        print(format_board(board))

        winner = check_winner(board)
        if winner == -1.0:
            print("\nOutcome: Random Player (O) won the game!")
            break
        if is_board_full(board):
            print("\nOutcome: Game ended in a Draw!")
            break

        turn += 1

    print("====================================================\n")


def main():
    # Number of training episodes (defaults to 1500; can be overridden via env var)
    total_episodes = int(os.environ.get("TIC_TAC_TOE_EPISODES", "1500"))

    # 1. Build policy neural network
    net = build_policy_network(learning_rate=0.01)

    # 2. Train policy via Reinforcement Learning
    train_agent(net, total_episodes=total_episodes)

    # 3. Evaluate trained policy against Random player over 100 matches
    evaluate_against_random(net, num_games=100)

    # 4. Play and display a step-by-step demonstration game
    play_demonstration_game(net)


if __name__ == "__main__":
    main()
