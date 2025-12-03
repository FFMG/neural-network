#pragma once

#include "activation.h"
#include "errorcalculation.h"
#include "hiddenstate.h"
#include "layer.h" // For LayerType enum
#include "weightparam.h"

#include <vector>

// Forward declaration to allow BaseLayer to be used in function signatures.
class BaseLayer;

/**
 * @class BaseLayer
 * @brief An abstract interface for a layer in a neural network.
 *
 * This class defines the common interface for all layer types, whether they are
 * part of a Feedforward (FNN) or Recurrent (RNN) Neural Network. It is designed
 * to be inherited by concrete layer implementations.
 */
class BaseLayer {
public:
  /**
   * @brief Virtual destructor to ensure proper cleanup of derived classes.
   */
  virtual ~BaseLayer() = default;

  // --- Core Layer Properties ---

  /**
   * @brief Gets the index of this layer within the network.
   * @return The zero-based index of the layer.
   */
  virtual unsigned get_layer_index() const noexcept = 0;

  /**
   * @brief Gets the type of the layer (Input, Hidden, or Output).
   * @return The LayerType enum value.
   */
  virtual Layer::LayerType layer_type() const = 0;

  /**
   * @brief Gets the number of neurons in this layer.
   * @return The total number of neurons.
   */
  virtual unsigned number_neurons() const noexcept = 0;

  /**
   * @brief Gets the number of input connections to this layer.
   * @param add_bias Whether to include the bias neuron in the count.
   * @return The number of input neurons from the previous layer.
   */
  virtual unsigned number_input_neurons(bool add_bias) const noexcept = 0;

  // --- Forward Pass ---

  /**
   * @brief Performs the forward pass calculation for a batch of inputs.
   * @param previous_layer A constant reference to the preceding layer in the
   * network.
   * @param previous_layer_inputs The output values from the previous layer.
   * @param residual_output_values The output values from a residual connection,
   * if any.
   * @param hidden_states A reference to the hidden states for storing
   * intermediate values (e.g., pre-activation sums) needed for backpropagation.
   * @param is_training A flag indicating whether the network is in training
   * mode.
   * @return A matrix of output values for the current layer.
   */
  virtual std::vector<std::vector<double>> calculate_forward_feed(
      const BaseLayer &previous_layer,
      const std::vector<std::vector<double>> &previous_layer_inputs,
      const std::vector<std::vector<double>> &residual_output_values,
      std::vector<std::vector<HiddenState>> &hidden_states,
      bool is_training) const = 0;

  // --- Backward Pass (Gradient Calculation) ---

  /**
   * @brief Calculates the gradients for the output layer.
   * @param target_outputs The expected or target values.
   * @param given_outputs The actual computed output values from the forward
   * pass.
   * @param hidden_states The hidden states recorded during the forward pass.
   * @param gradient_clip_threshold The threshold value for clipping gradients.
   * @param error_calculation_type The type of error calculation method to use.
   * @return A matrix of gradients for the output layer.
   */
  virtual std::vector<std::vector<double>> calculate_output_gradients(
      const std::vector<std::vector<double>> &target_outputs,
      const std::vector<std::vector<double>> &given_outputs,
      const std::vector<std::vector<HiddenState>> &hidden_states,
      double gradient_clip_threshold,
      ErrorCalculation::type error_calculation_type) const = 0;

  /**
   * @brief Calculates the gradients for a hidden layer.
   * @param next_layer A constant reference to the subsequent layer in the
   * network.
   * @param next_grad_matrix The gradient matrix from the next layer.
   * @param output_matrix The output matrix of the current layer from the
   * forward pass.
   * @param hidden_states The hidden states recorded during the forward pass.
   * @param gradient_clip_threshold The threshold value for clipping gradients.
   * @return A matrix of gradients for the hidden layer.
   */
  virtual std::vector<std::vector<double>> calculate_hidden_gradients(
      const BaseLayer &next_layer,
      const std::vector<std::vector<double>> &next_grad_matrix,
      const std::vector<std::vector<double>> &output_matrix,
      const std::vector<std::vector<HiddenState>> &hidden_states,
      double gradient_clip_threshold) const = 0;

  // --- Weights and Biases ---

  /**
   * @brief Gets a constant reference to the layer's weight parameters.
   * @return A 2D vector of WeightParam objects.
   */
  virtual const std::vector<std::vector<WeightParam>> &
  get_weight_params() const = 0;

  /**
   * @brief Gets a constant reference to the layer's bias weight parameters.
   * @return A vector of WeightParam objects for the bias weights.
   */
  virtual const std::vector<WeightParam> &get_bias_weight_params() const = 0;

  /**
   * @brief Checks if the layer has a bias neuron.
   * @return True if the layer includes a bias, false otherwise.
   */
  virtual bool has_bias() const noexcept = 0;

  // --- Activation Function ---

  /**
   * @brief Gets a constant reference to the layer's activation function logic.
   * @return The activation object.
   */
  virtual const activation &get_activation() const noexcept = 0;
};
