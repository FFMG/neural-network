# Reinforcement Learning Guide

This guide provides a comprehensive overview of Reinforcement Learning (RL) in the `myoddweb::nn` neural network library. It details the underlying policy-gradient architecture, all available configuration options and hyperparameters, mathematical formulations, invalid-action masking rules, and practical code examples in both C++ and Python.

---

## Table of Contents

1. [Overview & Core Architecture](#overview--core-architecture)
2. [Supervised Learning vs Reinforcement Learning](#supervised-learning-vs-reinforcement-learning)
3. [Mathematical Foundations](#mathematical-foundations)
   - [Policy Gradient Formulation (REINFORCE)](#policy-gradient-formulation-reinforce)
   - [Advantage Scaling](#advantage-scaling)
   - [Entropy Regularisation Bonus](#entropy-regularisation-bonus)
   - [Dual Temperature Scaling](#dual-temperature-scaling)
4. [Configuration Options Reference](#configuration-options-reference)
   - [NeuralNetworkOptions Parameters](#neuralnetworkoptions-parameters)
   - [Output Layer & Activation Settings](#output-layer--activation-settings)
   - [EvaluationConfig Settings](#evaluationconfig-settings)
   - [Runtime Temperature Controls](#runtime-temperature-controls)
5. [Invalid-Action Masking](#invalid-action-masking)
   - [Purpose and Mask Definition](#purpose-and-mask-definition)
   - [Mathematical Mechanics and Renormalisation](#mathematical-mechanics-and-renormalisation)
   - [Validation Constraints and Error Handling](#validation-constraints-and-error-handling)
6. [Practical Code Examples](#practical-code-examples)
   - [C++ Implementation](#c-implementation)
   - [Python Implementation](#python-implementation)
7. [Best Practices and Hyperparameter Tuning](#best-practices-and-hyperparameter-tuning)

---

## Overview & Core Architecture

The `myoddweb::nn` library provides an on-policy policy-gradient engine based on the **REINFORCE** algorithm with scalar advantage weighting. In this paradigm:

- The neural network acts as a **parameterised policy** $\pi_\theta(a \mid s)$, mapping state vectors $s \in \mathbb{R}^D$ to action probability distributions over a discrete action space $\{a_1, a_2, \dots, a_K\}$.
- During rollouts, an agent interacts with an environment, observing states and sampling actions according to $\pi_\theta$.
- At the conclusion of a sequence, episode, or decision step, an **advantage** $A \in \mathbb{R}$ is computed to quantify how much better (or worse) the selected action was compared to an expected baseline.
- The policy weights are updated using `train_with_advantages`, which applies advantage-scaled gradients directly through all upstream layers.

```
                      +-------------------+
                      |    Environment    |
                      +-------------------+
                        |               ^
               State s  |               | Action a
                        v               |
             +--------------------+     |
             |   Neural Network   |-----+
             |    (Softmax)       |
             +--------------------+
                        |
                        v
          Policy: \pi_\theta(a | s)
                        |
       Trajectory Rollout + Advantage A
                        |
                        v
    +---------------------------------------+
    | train_with_advantages(s, a, A, [mask])|
    +---------------------------------------+
```

---

## Supervised Learning vs Reinforcement Learning

The library supports two distinct training workflows:

| Aspect | Supervised Learning (`train`) | Reinforcement Learning (`train_with_advantages`) |
| :--- | :--- | :--- |
| **Objective** | Minimise empirical loss $\mathcal{L}(\hat{y}, y^*)$ against static ground-truth targets $y^*$. | Maximise expected cumulative reward $\mathbb{E}[R]$ via policy gradients. |
| **Target Data** | Ground-truth labels or continuous target values supplied in advance. | Self-generated actions encoded as one-hot vectors paired with scalar advantages $A$. |
| **Execution Cycle** | Multi-epoch iteration over fixed datasets with early stopping, validation checks, and learning rate scheduling. | Single-pass on-policy update over transient trajectory steps collected online. |
| **Error Calculation** | Standard loss functions (MSE, MAE, Cross-Entropy, Huber, Quantile Loss, etc.). | Gradient scaled by advantage: $\delta = A \cdot (\hat{y} - y)$. |
| **Optimiser State** | Persists across epochs within the training loop. | Persists indefinitely across successive online environment updates. |

---

## Mathematical Foundations

### Policy Gradient Formulation (REINFORCE)

The objective of reinforcement learning is to find policy parameters $\theta$ that maximise expected cumulative return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

According to the Policy Gradient Theorem, the gradient of this objective with respect to $\theta$ is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t \mid s_t) \cdot A_t \right]$$

When the network's output layer uses the **Softmax** activation function with **Cross-Entropy** loss, the analytical gradient of the cross-entropy loss with respect to pre-activation logit $z_k$ is:

$$\frac{\partial \mathcal{L}_{CE}}{\partial z_k} = p_k - y_k$$

where $p_k = \pi_\theta(a_k \mid s)$ is the predicted probability for action $k$, and $y_k$ is the target indicator ($y_k = 1.0$ for the chosen action, $0.0$ otherwise).

### Advantage Scaling

The method `train_with_advantages` applies scalar advantage weighting $A_t$ directly to the output gradient delta $\delta_k$:

$$\delta_k = A_t \cdot (p_k - y_k)$$

For the chosen action $a$ where $y_a = 1.0$:

$$\delta_a = A_t \cdot (p_a - 1.0) = -A_t \cdot (1.0 - p_a)$$

For unselected actions $j \neq a$ where $y_j = 0.0$:

$$\delta_j = A_t \cdot p_j$$

During gradient descent, parameters are adjusted in the opposite direction of the gradient ($w \leftarrow w - \eta \delta$):

- **Positive Advantage ($A_t > 0$)**: $\delta_a < 0$, which causes gradient descent to increase the weight contributions to logit $z_a$, reinforcing the chosen action and increasing $p_a$.
- **Negative Advantage ($A_t < 0$)**: $\delta_a > 0$, which causes gradient descent to decrease the logit $z_a$, penalising the chosen action and reducing $p_a$.
- **Zero Advantage ($A_t = 0$)**: $\delta_k = 0$ for all $k$, preserving current weights without modification.

### Entropy Regularisation Bonus

In reinforcement learning, policies can prematurely collapse towards deterministic, sub-optimal actions before discovering superior strategies. To sustain adequate exploration, the library supports policy entropy regularisation parameterised by the entropy coefficient $\beta$ (`with_entropy_coefficient`).

The Shannon entropy of policy distribution $p$ is:

$$H(p) = -\sum_{j=1}^{K} p_j \ln p_j$$

The augmented training objective becomes:

$$J_{aug}(\theta) = J(\theta) + \beta H(\pi_\theta)$$

To maximise entropy alongside reward, the library adds the negative gradient of entropy to the backpropagation deltas:

$$\Delta \delta_k = \frac{\partial (-\beta H)}{\partial z_k} = \beta p_k \left( \ln p_k + H(p) \right)$$

- When an action is overly dominant, $\ln p_k$ is high and entropy is low, resulting in a positive adjustment that pulls probability away from the dominant mode towards a uniform distribution.
- When $\beta = 0.0$, entropy regularisation is disabled.
- Entropy regularisation operates exclusively on `Softmax` output layers.

### Dual Temperature Scaling

Softmax activation supports two independent temperature parameters to cleanly decouple exploration during training from deterministic execution during deployment:

$$p_k = \frac{\exp\left(\frac{z_k}{T}\right)}{\sum_j \exp\left(\frac{z_j}{T}\right)}$$

1. **Training Temperature ($T_{\text{train}}$)**: Controls logit smoothing when interacting with the environment during training. Setting $T_{\text{train}} > 1.0$ softens the distribution, encouraging stochastic exploration across alternate actions.
2. **Inference Temperature ($T_{\text{infer}}$)**: Controls logit sharpness during evaluation and production deployment. Setting $T_{\text{infer}} < 1.0$ (or approaching $0.0$) sharpens probabilities towards greedy selection.
3. **Logit Capping (`logit_cap`)**: Bounds logits within $[-C, C]$ before exponentiation to prevent numerical overflow and saturated gradients.

---

## Configuration Options Reference

### NeuralNetworkOptions Parameters

When configuring a model for reinforcement learning via `NeuralNetworkOptions`, the following options are relevant:

| Option | Method Name | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Entropy Coefficient** | `with_entropy_coefficient` | `double` | `0.0` | Weight $\beta \ge 0.0$ for policy entropy bonus. Values between $0.001$ and $0.1$ promote exploration. |
| **Learning Rate** | `with_learning_rate` | `double` | `0.01` | Optimisation step size $\eta$. Policy gradients typically perform best with rates in $[0.0005, 0.01]$. |
| **Batch Size** | `with_batch_size` | `size_t` | `1` | Number of trajectory steps per optimiser weight update chunk. |
| **Gradient Clipping** | `with_clip_threshold` | `double` | `0.0` (off) | Maximum gradient norm threshold. Useful to prevent divergence caused by high-variance advantage estimates. |
| **Hidden Layers** | `with_hidden_layers` | `std::vector<LayerDetails>` | Empty | Sequence of feature extraction layers (FeedForward, GRN, LSTM, Attention, etc.). |
| **Output Layer Details** | `with_output_layer_details` | `OutputLayerDetails` | Mandatory | Specification of the policy head. **Must be a single output layer**; multi-output branched heads are not supported for `train_with_advantages`. |
| **Bias** | `with_has_bias` | `bool` | `true` | Enables learnable bias units for each neuron layer. |
| **Random Seed** | `with_seed` | `std::optional<uint32_t>` | `nullopt` | Seed for weight initialisation and reproducible experimentation. |

### Output Layer & Activation Settings

For discrete action spaces, the output layer must be configured as follows:

```cpp
OutputLayerDetails(
  number_of_actions,                                   // Number of discrete actions
  activation(activation::method::softmax, 0.0, 1.0),   // Softmax activation
  ErrorCalculation::type::cross_entropy,               // Cross-Entropy loss
  EvaluationConfig(),                                  // Metric evaluation config
  0.0,                                                 // Weight decay (optional)
  OptimiserType::Adam,                                 // Optimiser (e.g. Adam, AdamW, RMSProp)
  0.9                                                  // Momentum parameter
);
```

#### Activation Constructor Signatures

- Standard:
  ```cpp
  activation(activation::method::softmax, double alpha = 0.0, double temperature = 1.0);
  ```
- Dual Temperature & Capping:
  ```cpp
  activation(activation::method::softmax, double alpha, double temperature, double inference_temperature, double logit_cap = 0.0);
  ```

### EvaluationConfig Settings

The `EvaluationConfig` struct controls forecast and metric calculations:

- `with_cross_entropy_lambda(double lambda)`: Multiplier weighting for cross-entropy loss computation.
- `with_confidence_threshold(double threshold)`: Minimum prediction probability threshold for ratio and coverage metrics.

### Runtime Temperature Controls

Temperatures can be inspected and updated dynamically at runtime without reconstructing the model:

```cpp
// Set inference temperature for greedy evaluation:
nn.set_inference_temperature(0.1);

// Inspect active temperatures:
double train_temp = nn.get_temperature();
double infer_temp = nn.get_inference_temperature();

// Revert or adjust training temperature:
nn.set_temperature(1.2);
```

In Python:
```python
net.set_inference_temperature(0.1)
train_temp = net.get_temperature()
infer_temp = net.get_inference_temperature()
```

---

## Invalid-Action Masking

### Purpose and Mask Definition

In many reinforcement learning environments, certain actions are physically or strategically impossible in specific states (for example, placing a piece in an already occupied square in Tic-Tac-Toe, or submitting an invalid order).

Invalid-action masking prevents the policy from learning illegal actions and ensures that negative advantage updates do not inadvertently redistribute probability onto forbidden options.

An action mask is provided as a vector matching the action space size:
- `1.0`: The action is **legal** (valid).
- `0.0`: The action is **illegal** (invalid).

### Mathematical Mechanics and Renormalisation

When `train_with_advantages(states, actions, advantages, action_masks)` is invoked:

1. **Legal Probability Renormalisation**:
   The predicted probability distribution $p$ is renormalised across the legal subset $\mathcal{M} = \{j \mid m_j = 1.0\}$:
   $$q_k = \frac{p_k}{\sum_{j \in \mathcal{M}} p_j} \quad \forall k \in \mathcal{M}$$
2. **Zero-Gradient Masking for Illegal Actions**:
   For any illegal action $i \notin \mathcal{M}$, the target delta is set to match the model's raw probability ($t'_i = p_i$), producing an error delta of zero:
   $$\delta_i = A_t \cdot (p_i - t'_i) = 0.0$$
   Consequently, weights associated with illegal actions receive no gradient update.
3. **Redistribution Across Legal Alternatives**:
   When an action receives a negative advantage ($A_t < 0$), the penalty reduces the chosen action's probability and increases the probabilities of remaining actions. The mask ensures that this redistributed probability shifts **strictly onto other legal actions**, never onto invalid ones.
4. **Masked Entropy Bonus**:
   If `entropy_coefficient` $\beta > 0$, entropy is computed exclusively over the renormalised legal distribution $q$:
   $$H_{\text{legal}} = -\sum_{j \in \mathcal{M}} q_j \ln q_j$$
   The resulting exploration bonus pushes towards uniform exploration across legal moves only.

### Validation Constraints and Error Handling

The library performs strict validation on action masks:

1. **Softmax Output Requirement**: Masking is supported exclusively on output layers utilising `activation::method::softmax`. Attempting to use masks with linear or other activations throws `std::runtime_error`.
2. **Binary Value Constraint**: Every mask entry must be either `0.0` or `1.0`. Any other floating-point value throws `std::runtime_error`.
3. **At Least One Legal Action**: Every sample in the batch must have at least one legal action ($\sum_j m_j \ge 1$). A mask where all actions are `0.0` throws `std::runtime_error`.
4. **No Illegal Selections**: The one-hot action target for any illegal action must be `0.0`. If an action target indicates an illegal move ($y_k = 1.0$ where $m_k = 0.0$), `std::runtime_error` is thrown.
5. **Dimensional Conformity**: Mask vector dimensions must match `states.size()` along the batch axis and `output_layer_size` along the action axis.
6. **No Recurrent BPTT Support**: Action masks cannot be combined with Backpropagation Through Time (`enable_bptt(true)`).

---

## Practical Code Examples

### C++ Implementation

The following complete example demonstrates configuring a policy network, collecting trajectory transitions with action masking, calculating discounted advantages, and performing an on-policy update.

```cpp
#include "neuralnetwork.h"
#include "neuralnetworkoptions.h"
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>

using namespace myoddweb::nn;

// Selects an action using roulette-wheel sampling over valid moves
size_t select_masked_action(
    const std::vector<double>& probabilities,
    const std::vector<double>& legal_mask,
    std::mt19937& rng)
{
    std::vector<double> filtered_probs(probabilities.size(), 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < probabilities.size(); ++i)
    {
        if (legal_mask[i] > 0.5)
        {
            filtered_probs[i] = probabilities[i];
            sum += probabilities[i];
        }
    }

    if (sum <= 1e-12)
    {
        // Fallback uniform selection among legal actions
        std::vector<size_t> legal_indices;
        for (size_t i = 0; i < legal_mask.size(); ++i)
        {
            if (legal_mask[i] > 0.5)
            {
                legal_indices.push_back(i);
            }
        }
        std::uniform_int_distribution<size_t> dist(0, legal_indices.size() - 1);
        return legal_indices[dist(rng)];
    }

    std::uniform_real_distribution<double> dist(0.0, sum);
    double roll = dist(rng);
    double accumulated = 0.0;

    for (size_t i = 0; i < filtered_probs.size(); ++i)
    {
        if (legal_mask[i] > 0.5)
        {
            accumulated += filtered_probs[i];
            if (roll <= accumulated)
            {
                return i;
            }
        }
    }

    return 0;
}

int main()
{
    const unsigned state_dim = 4;
    const unsigned action_dim = 3;
    const double learning_rate = 0.01;
    const double discount_factor = 0.95;

    // 1. Configure the policy network
    std::vector<LayerDetails> hidden_layers =
    {
        LayerDetails(
            Layer::Architecture::FF,
            16,
            activation(activation::method::relu, 0.0),
            0.0,                    // Dropout
            0.0001,                 // Weight decay
            OptimiserType::AdamW,
            0.9,
            false, 0, 0, 0, 0, 0, 0, 0)
    };

    EvaluationConfig eval_config;
    OutputLayerDetails output_details(
        action_dim,
        activation(activation::method::softmax, 0.0, /*training_temp=*/1.0, /*inference_temp=*/0.2),
        ErrorCalculation::type::cross_entropy,
        eval_config,
        0.0001,
        OptimiserType::AdamW,
        0.9);

    auto options = NeuralNetworkOptions::create({ state_dim, 16, action_dim })
        .with_hidden_layers(hidden_layers)
        .with_output_layer_details(output_details)
        .with_learning_rate(learning_rate)
        .with_batch_size(16)
        .with_entropy_coefficient(0.02)  // Sustains exploration
        .with_has_bias(true)
        .with_seed(42u)
        .build();

    NeuralNetwork policy_net(options);

    std::mt19937 rng(42);

    // 2. Collect trajectory data across an episode
    std::vector<std::vector<double>> trajectory_states;
    std::vector<std::vector<double>> trajectory_actions;
    std::vector<std::vector<double>> trajectory_masks;
    std::vector<double> trajectory_rewards;

    // Simulated 3-step episode
    for (size_t step = 0; step < 3; ++step)
    {
        std::vector<double> current_state = { 0.5, -0.2, 0.1 * step, 0.8 };
        std::vector<double> legal_mask = { 1.0, 1.0, (step == 2 ? 0.0 : 1.0) };

        // Query model probabilities
        std::vector<double> action_probs = policy_net.think(current_state);

        // Sample action
        size_t chosen_action = select_masked_action(action_probs, legal_mask, rng);

        // Build one-hot target
        std::vector<double> action_target(action_dim, 0.0);
        action_target[chosen_action] = 1.0;

        // Simulated environment reward
        double reward = (chosen_action == 1) ? 1.0 : -0.5;

        trajectory_states.push_back(current_state);
        trajectory_actions.push_back(action_target);
        trajectory_masks.push_back(legal_mask);
        trajectory_rewards.push_back(reward);
    }

    // 3. Compute discounted returns as advantages
    const size_t T = trajectory_rewards.size();
    std::vector<double> advantages(T, 0.0);
    double running_return = 0.0;

    for (int t = static_cast<int>(T) - 1; t >= 0; --t)
    {
        running_return = trajectory_rewards[t] + discount_factor * running_return;
        advantages[t] = running_return;
    }

    // 4. Update the policy using advantage scaling and action masking
    policy_net.train_with_advantages(
        trajectory_states,
        trajectory_actions,
        advantages,
        trajectory_masks);

    std::cout << "Policy update completed successfully.\n";

    // 5. Evaluate policy greedily at low inference temperature
    policy_net.set_inference_temperature(0.05);
    std::vector<double> eval_state = { 0.5, -0.2, 0.2, 0.8 };
    std::vector<double> greedy_probs = policy_net.think(eval_state);

    std::cout << "Evaluated action probabilities: ";
    for (double p : greedy_probs)
    {
        std::cout << p << " ";
    }
    std::cout << "\n";

    return 0;
}
```

---

### Python Implementation

The following complete script demonstrates the same workflow in Python using the `neuralnetwork` package.

```python
import os
import sys
import random

try:
    import neuralnetwork as nn
except ImportError:
    # Add library search path if uninstalled
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "x64", "Release")))
    import neuralnetwork as nn


def build_agent():
    state_dim = 4
    hidden_dim = 16
    action_dim = 3

    # Define hidden layer with AdamW optimiser
    hidden_layers = [
        nn.LayerDetails(
            nn.LayerArchitecture.FF,
            hidden_dim,
            nn.Activation(nn.ActivationMethod.Relu, 0.0),
            0.0,                 # Dropout rate
            0.0001,              # Weight decay
            nn.OptimiserType.AdamW,
            0.9                  # Momentum
        )
    ]

    # Softmax output head with dual temperature scaling
    output_layer = nn.OutputLayerDetails(
        action_dim,
        nn.Activation(nn.ActivationMethod.Softmax, 0.0, 1.0, 0.2), # train_temp=1.0, infer_temp=0.2
        nn.ErrorCalculationType.CrossEntropy,
        nn.EvaluationConfig(),
        0.0001,
        nn.OptimiserType.AdamW,
        0.9
    )

    options = (
        nn.NeuralNetworkOptions.create([state_dim, hidden_dim, action_dim])
        .with_hidden_layers(hidden_layers)
        .with_output_layer_details(output_layer)
        .with_learning_rate(0.01)
        .with_batch_size(16)
        .with_entropy_coefficient(0.02)  # Exploration bonus
        .with_has_bias(True)
        .with_seed(42)
        .build()
    )

    return nn.NeuralNetwork(options)


def sample_action(probabilities, legal_mask):
    """Samples an action from the renormalised legal distribution."""
    legal_indices = [i for i, m in enumerate(legal_mask) if m > 0.5]
    masked_probs = [probabilities[i] if i in legal_indices else 0.0 for i in range(len(probabilities))]
    total = sum(masked_probs)

    if total <= 1e-12:
        return random.choice(legal_indices)

    norm_probs = [p / total for p in masked_probs]
    roll = random.random()
    cumulative = 0.0
    for idx in legal_indices:
        cumulative += norm_probs[idx]
        if roll <= cumulative:
            return idx
    return legal_indices[-1]


def main():
    agent = build_agent()
    discount_factor = 0.95

    # Storage for batched updates
    batch_states = []
    batch_actions = []
    batch_advantages = []
    batch_masks = []

    # Simulate 50 episodes
    for episode in range(50):
        episode_states = []
        episode_actions = []
        episode_masks = []
        episode_rewards = []

        # Simulate 5 steps per episode
        for step in range(5):
            state = [random.uniform(-1.0, 1.0) for _ in range(4)]
            # Action 2 is illegal on the final step
            mask = [1.0, 1.0, 0.0 if step == 4 else 1.0]

            probs = agent.think(state)
            action = sample_action(probs, mask)

            one_hot_action = [0.0] * 3
            one_hot_action[action] = 1.0

            # Environment reward
            reward = 1.0 if action == 1 else -0.2

            episode_states.append(state)
            episode_actions.append(one_hot_action)
            episode_masks.append(mask)
            episode_rewards.append(reward)

        # Compute discounted returns
        T = len(episode_rewards)
        returns = [0.0] * T
        running_return = 0.0
        for t in reversed(range(T)):
            running_return = episode_rewards[t] + discount_factor * running_return
            returns[t] = running_return

        batch_states.extend(episode_states)
        batch_actions.extend(episode_actions)
        batch_advantages.extend(returns)
        batch_masks.extend(episode_masks)

        # Batch update policy
        if len(batch_states) >= 16:
            agent.train_with_advantages(
                batch_states,
                batch_actions,
                batch_advantages,
                batch_masks
            )
            batch_states.clear()
            batch_actions.clear()
            batch_advantages.clear()
            batch_masks.clear()

    print("Reinforcement Learning training complete.")

    # Evaluate greedily using deployment inference temperature
    agent.set_inference_temperature(0.01)
    test_state = [0.2, -0.5, 0.1, 0.9]
    eval_probs = agent.think(test_state)
    print(f"Greedy inference probabilities: {[round(p, 4) for p in eval_probs]}")


if __name__ == "__main__":
    main()
```

---

## Best Practices and Hyperparameter Tuning

1. **Advantage Normalisation**:
   When training over diverse or long episodes, raw cumulative returns can exhibit high variance. Subtracting the batch mean and dividing by the standard deviation stabilizes training:
   $$\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + 10^{-8}}$$
2. **Entropy Tuning ($\beta$)**:
   - Begin with $\beta \in [0.01, 0.05]$.
   - If the agent collapses into picking a single action prematurely, increase $\beta$.
   - If the agent acts randomly and fails to converge to an optimal policy, reduce $\beta$.
3. **Decoupled Temperature Tuning**:
   Keep $T_{\text{train}} = 1.0$ (or slightly higher, e.g. $1.1 - 1.3$) to encourage exploration during training rollouts. For final deployment or greedy play, lower $T_{\text{infer}}$ to $0.05 - 0.2$.
4. **Optimiser Selection**:
   `AdamW` or `Adam` with momentum $0.9$ generally delivers superior stability for policy gradients compared to plain `SGD`.
5. **Gradient Clipping**:
   Enable `with_clip_threshold(1.0)` or `2.0` when environment rewards have high variance or rare extreme spikes to protect weights from divergence.
6. **Action Masking**:
   Always pass valid action masks during both action sampling and the `train_with_advantages` call. This prevents invalid options from corrupting policy gradients or absorbing probability mass.
