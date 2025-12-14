#pragma once

#include "./libraries/instrumentor.h"

#include "activation.h"
#include "errorcalculation.h"
#include "gradientsandoutputs.h"

#include "hiddenstate.h"
#include "weightparam.h"

#include <vector>

// Forward declaration to allow Layer to be used in function signatures.
class Layer;

/**
 * @class Layer
 * @brief An abstract interface for a layer in a neural network.
 *
 * This class defines the common interface for all layer types, whether they are
 * part of a Feedforward (FNN) or Recurrent (RNN) Neural Network. It is designed
 * to be inherited by concrete layer implementations.
 */
class Layer {
public:
  enum class LayerType
  {
    Input,
    Hidden,
    Output,
    Recurrent
  };

  Layer(
    unsigned layer_index,
    LayerType layer_type,
    const activation::method& activation_method,
    unsigned number_input_neurons,
    unsigned number_output_neurons
    ) noexcept :
    _layer_index(layer_index),
    _layer_type(layer_type),
    _activation(activation_method),
    _number_input_neurons(number_input_neurons),
    _number_output_neurons(number_output_neurons)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
  }

  Layer(const Layer& src) noexcept :
    _layer_index(src._layer_index),
    _layer_type(src._layer_type),
    _activation(src._activation),
    _number_input_neurons(src._number_input_neurons),
    _number_output_neurons(src._number_output_neurons)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
  }

  Layer(Layer&& src) noexcept :
    _layer_index(src._layer_index),
    _layer_type(src._layer_type),
    _activation(std::move(src._activation)),
    _number_input_neurons(src._number_input_neurons),
    _number_output_neurons(src._number_output_neurons)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    src._layer_type = LayerType::Input;
    src._layer_index = 0;
    src._number_input_neurons = 0;
    src._number_output_neurons = 0;
  }

  Layer& operator=(const Layer& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (this != &src)
    {
      _layer_index = src._layer_index;
      _layer_type = src._layer_type;
      _activation = src._activation;
      _number_input_neurons = src._number_input_neurons;
      _number_output_neurons = src._number_output_neurons;
    }
    return *this;
  }

  Layer& operator=(Layer&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (this != &src)
    {
      _layer_index = src._layer_index;
      _layer_type = src._layer_type;
      _activation = std::move(src._activation);
      _number_input_neurons = src._number_input_neurons;
      _number_output_neurons = src._number_output_neurons;
      
      src._layer_index = 0;
      src._number_input_neurons = 0;
      src._number_output_neurons = 0;
    }
    return *this;
  }

  virtual ~Layer() = default;

  // --- Core Layer Properties ---

  inline unsigned get_layer_index() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_index;
  }

  inline LayerType get_layer_type() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_type;
  }

  virtual int residual_layer_number() const = 0;

  inline unsigned number_neurons() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return get_number_output_neurons();
  }

  inline unsigned get_number_input_neurons() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _number_input_neurons;
  }

  inline unsigned get_number_output_neurons() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _number_output_neurons;
  }

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
  virtual std::vector<double> calculate_forward_feed(
      GradientsAndOutputs& gradients_and_outputs,
      const Layer &previous_layer,
      const std::vector<double> &previous_layer_inputs,
      const std::vector<double> &residual_output_values,
      std::vector<HiddenState> &hidden_states,
      bool is_training) const = 0;

  // --- Backward Pass (Gradient Calculation) ---

  virtual void calculate_output_gradients(
      GradientsAndOutputs& gradients_and_outputs,
      const std::vector<double> &target_outputs,
      const std::vector<HiddenState> &hidden_states,
      double gradient_clip_threshold,
      ErrorCalculation::type error_calculation_type) const = 0;

  const activation& get_activation() const noexcept
  {
    return _activation;
  }

  virtual void calculate_hidden_gradients(
      GradientsAndOutputs& gradients_and_outputs,
      const Layer &next_layer,
      const std::vector<double> &next_grad_matrix,
      const std::vector<double> &output_matrix,
      const std::vector<HiddenState> &hidden_states,
      double gradient_clip_threshold) const = 0;

  virtual void apply_weight_gradient(double gradient, double learning_rate, bool is_bias, WeightParam& weight_param, double clipping_scale, double gradient_clip_threshold) = 0;

  // --- Weights and Biases ---

  /**
   * @brief Gets a constant reference to the layer's weight parameters.
   * @return A 2D vector of WeightParam objects.
   */
  virtual const std::vector<std::vector<WeightParam>> &get_weight_params() const = 0;

  /**
   * @brief Gets a specific weight parameter from the layer.
   * @param input_neuron_number The index of the input neuron.
   * @param neuron_index The index of the current neuron.
   * @return A constant reference to the WeightParam.
   */
  virtual const WeightParam& get_weight_param(unsigned input_neuron_number, unsigned neuron_index) const = 0;

  /**
   * @brief Gets a specific weight parameter from the layer (mutable).
   * @param input_neuron_number The index of the input neuron.
   * @param neuron_index The index of the current neuron.
   * @return A mutable reference to the WeightParam.
   */
  virtual WeightParam& get_weight_param(unsigned input_neuron_number, unsigned neuron_index) = 0;

  /**
   * @brief Gets a constant reference to the layer's bias weight parameters.
   * @return A vector of WeightParam objects for the bias weights.
   */
  virtual const std::vector<WeightParam> &get_bias_weight_params() const = 0;
  virtual WeightParam& get_bias_weight_param(unsigned neuron_index) = 0;

  virtual const std::vector<std::vector<WeightParam>>& get_residual_weight_params() const = 0;
  virtual std::vector<std::vector<WeightParam>>& get_residual_weight_params() = 0;
  virtual std::vector<WeightParam>& get_residual_weight_params(unsigned neuron_index) = 0;


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

  /**
   * @brief Clones the layer, creating a deep copy.
   * @return A pointer to the newly created Layer.
   */
  virtual Layer* clone() const = 0;

private:
  unsigned _layer_index;
  LayerType _layer_type;
  activation _activation;

  unsigned _number_input_neurons;  //  number of neurons in previous layer
  unsigned _number_output_neurons; //  number of neurons in this layer
};