#pragma once

#include "./libraries/instrumentor.h"

#include "activation.h"
#include "errorcalculation.h"
#include "gradientsandoutputs.h"
#include "hiddenstate.h"
#include "neuron.h"
#include "optimiser.h"
#include "residualprojector.h"
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
class Layer
{
public:
  enum class LayerType
  {
    Input,
    Hidden,
    Output
  };

protected:
  Layer(
    unsigned layer_index,
    LayerType layer_type,
    const activation::method& activation_method,
    OptimiserType optimiser_type,
    int residual_layer_number,
    unsigned number_input_neurons,
    unsigned number_output_neurons,
    const std::vector<Neuron>& neurons,
    const std::vector<std::vector<WeightParam>>& weights,
    const std::vector<WeightParam>& bias_weights,
    ResidualProjector* residual_projector
  ) noexcept :
    _layer_index(layer_index),
    _layer_type(layer_type),
    _activation(activation_method),
    _optimiser_type(optimiser_type),
    _residual_layer_number(residual_layer_number),
    _number_input_neurons(number_input_neurons),
    _number_output_neurons(number_output_neurons),
    _neurons(neurons),
    _weights(weights),
    _bias_weights(bias_weights),
    _residual_projector(residual_projector)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (number_output_neurons == 0)
    {
      Logger::panic("Error: Creating a layer with 0 neurons.");
    }
    if (layer_type != LayerType::Input && number_input_neurons == 0)
    {
      Logger::warning("Warning: Non-input layer created with 0 inputs.");
    }
  }

public:
  Layer(const Layer& src) noexcept :
    _layer_index(src._layer_index),
    _layer_type(src._layer_type),
    _activation(src._activation),
    _optimiser_type(src._optimiser_type),
    _residual_layer_number(src._residual_layer_number),
    _number_input_neurons(src._number_input_neurons),
    _number_output_neurons(src._number_output_neurons),
    _neurons(src._neurons),
    _weights(src._weights),
    _bias_weights(src._bias_weights),
    _residual_projector(src._residual_projector)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (src._residual_projector != nullptr)
    {
      _residual_projector = new ResidualProjector(*src._residual_projector);
    }
  }

  Layer(Layer&& src) noexcept :
    _layer_index(src._layer_index),
    _layer_type(src._layer_type),
    _activation(std::move(src._activation)),
    _optimiser_type(std::move(src._optimiser_type)),
    _number_input_neurons(src._number_input_neurons),
    _number_output_neurons(src._number_output_neurons),
    _neurons(std::move(src._neurons)),
    _weights(std::move(src._weights)),
    _bias_weights(std::move(src._bias_weights)),
    _residual_layer_number(src._residual_layer_number),
    _residual_projector(src._residual_projector)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    src._layer_type = LayerType::Input;
    src._layer_index = 0;
    src._optimiser_type = OptimiserType::None;
    src._number_input_neurons = 0;
    src._number_output_neurons = 0;
    src._residual_layer_number = 0;
    src._residual_projector = nullptr;
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
      _neurons = src._neurons;
      _weights = src._weights;
      _bias_weights = src._bias_weights;
      _residual_layer_number = src._residual_layer_number;
      delete _residual_projector;
      _residual_projector = nullptr;
      if (src._residual_projector != nullptr)
      {
        _residual_projector = new ResidualProjector(*src._residual_projector);
      }
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
      _optimiser_type = std::move(src._optimiser_type);
      _number_input_neurons = src._number_input_neurons;
      _number_output_neurons = src._number_output_neurons;
      _neurons = std::move(src._neurons);
      _weights = std::move(src._weights);
      _bias_weights = std::move(src._bias_weights);
      delete _residual_projector;
      _residual_projector = src._residual_projector;

      src._layer_index = 0;
      src._optimiser_type = OptimiserType::None;
      src._number_input_neurons = 0;
      src._number_output_neurons = 0;
      src._residual_projector = nullptr;
    }
    return *this;
  }

  virtual ~Layer()
  {
    delete _residual_projector;
    _residual_projector = nullptr;
  }


  // --- Core Layer Properties ---

  inline OptimiserType get_optimiser_type() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _optimiser_type;
  }

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

  inline int get_residual_layer_number() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _residual_layer_number;
  }

  const ResidualProjector* get_residual_projector() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _residual_projector;
  }

  ResidualProjector* get_residual_projector()
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _residual_projector;
  }

  inline unsigned get_number_neurons() const noexcept
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

  virtual bool use_bptt() const noexcept {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return false;
  }

  virtual std::vector<double> calculate_forward_feed(
    GradientsAndOutputs& gradients_and_outputs,
    const Layer& previous_layer,
    const std::vector<double>& previous_layer_inputs,
    const std::vector<double>& residual_output_values,
    std::vector<HiddenState>& hidden_states,
    bool is_training) const = 0;

  virtual void calculate_output_gradients(
    GradientsAndOutputs& gradients_and_outputs,
    const std::vector<double>& target_outputs,
    const std::vector<HiddenState>& hidden_states,
    ErrorCalculation::type error_calculation_type) const = 0;

  inline const activation& get_activation() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _activation;
  }

  virtual void calculate_hidden_gradients(
    GradientsAndOutputs& gradients_and_outputs,
    const Layer& next_layer,
    const std::vector<double>& next_grad_matrix,
    const std::vector<double>& output_matrix,
    const std::vector<HiddenState>& hidden_states,
    int bptt_max_ticks) const = 0;

  virtual void apply_weight_gradient(double gradient, double learning_rate, bool is_bias, WeightParam& weight_param, double clipping_scale)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");

    if (!std::isfinite(gradient))
    {
      Logger::panic("Error while calculating input weigh gradient it invalid.");
    }
    auto old_velocity = weight_param.get_velocity();
    if (!std::isfinite(old_velocity))
    {
      Logger::panic("Error while calculating input weigh old velocity is invalid.");
    }

    if (clipping_scale < 0.0)
    {
      // If clipping scale is negative, we clip the gradient to a fixed range
      // This is useful for debugging or when we want to ensure gradients are not too large.
      Logger::warning("Clipping gradient to a fixed range.");
    }

    double final_gradient = gradient * clipping_scale;
    switch (_optimiser_type)
    {
    case OptimiserType::None:
      apply_none_update(weight_param, final_gradient, learning_rate);
      break;

    case OptimiserType::SGD:
      apply_sgd_update(weight_param, final_gradient, learning_rate, _activation.momentum(), is_bias);
      break;

    case OptimiserType::Adam:
      apply_adam_update(weight_param, final_gradient, learning_rate, 0.9, 0.999, 1e-8, is_bias);
      break;

    case OptimiserType::AdamW:
      apply_adamw_update(weight_param, final_gradient, learning_rate, 0.9, 0.999, 1e-8);
      break;

    case OptimiserType::Nadam:
      apply_nadam_update(weight_param, final_gradient, learning_rate, 0.9, 0.999, 1e-8);
      break;

    case OptimiserType::NadamW:
      apply_nadamw_update(weight_param, final_gradient, learning_rate, 0.9, 0.999, 1e-8, is_bias);
      break;

    default:
      Logger::panic("Unknown optimizer type:", (int)_optimiser_type);
    }
  }

  const std::vector<Neuron>& get_neurons() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _neurons;
  }

  const Neuron& get_neuron(unsigned int neuron_index) const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
#if VALIDATE_DATA == 1
    if (neuron_index >= _neurons.size())
    {
      Logger::panic("Index out of bounds in Layer::get_neuron.");
    }
#endif
    return _neurons[neuron_index];
  }

  // --- Weights and Biases ---

  const std::vector<std::vector<WeightParam>>& get_weight_params() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _weights;
  }

  const std::vector<WeightParam>& get_weight_param(unsigned int input_neuron_number) const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _weights[input_neuron_number];
  }

  const WeightParam& get_weight_param(unsigned int input_neuron_number, unsigned int neuron_index) const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _weights[input_neuron_number][neuron_index];
  }

  inline WeightParam& get_weight_param(unsigned int input_neuron_number, unsigned int neuron_index)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _weights[input_neuron_number][neuron_index];
  }

  inline const std::vector<WeightParam>& get_bias_weight_params() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _bias_weights;
  }

  inline const WeightParam& get_bias_weight_param(unsigned int neuron_index) const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _bias_weights[neuron_index];
  }

  inline WeightParam& get_bias_weight_param(unsigned int neuron_index)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _bias_weights[neuron_index];
  }

  virtual const std::vector<std::vector<WeightParam>>& get_residual_weight_params() const = 0;
  virtual std::vector<std::vector<WeightParam>>& get_residual_weight_params() = 0;

  /**
   * @brief Checks if the layer has a bias neuron.
   * @return True if the layer includes a bias, false otherwise.
   */
  virtual bool has_bias() const noexcept = 0;

  virtual Layer* clone() const = 0;

protected:

  static std::vector<std::vector<WeightParam>> create_weights(
    double weight_decay,
    LayerType layer_type,
    const activation& activation_method,
    unsigned number_input_neurons,
    unsigned number_output_neurons
  )
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
#if VALIDATE_DATA == 1
    if (number_input_neurons == 0)
    {
      assert(layer_type == LayerType::Input);
      return {};
    }
#endif

    std::vector<std::vector<WeightParam>> weights;
    weights.resize(number_input_neurons);
    for (unsigned i = 0; i < number_input_neurons; ++i)
    {
      auto inner_weights = activation_method.weight_initialization(number_output_neurons, number_input_neurons);
      assert(inner_weights.size() == number_output_neurons);
      weights[i].reserve(number_output_neurons);
      for (unsigned o = 0; o < number_output_neurons; ++o)
      {
        const auto& weight = inner_weights[o];
        weights[i].emplace_back(WeightParam(weight, 0.0, 0.0, weight_decay));
      }
    }
    return weights;
  }

  static std::vector<WeightParam> create_bias_weights(
    bool has_bias,
    const activation& activation_method,
    unsigned number_output_neurons
  )
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (!has_bias)
    {
      return {};
    }

    std::vector<WeightParam> bias_weights;
    bias_weights.reserve(number_output_neurons);
    auto weights = activation_method.weight_initialization(number_output_neurons, 1);
    for (unsigned o = 0; o < number_output_neurons; ++o)
    {
      const auto& weight = weights[o];
      bias_weights.emplace_back(WeightParam(weight, 0.0, 0.0, 0.0));
    }
    return bias_weights;
  }

  static std::vector<Neuron> create_neurons(
    double dropout_rate,
    unsigned number_output_neurons
  )
  {
    std::vector<Neuron> neurons;
    neurons.reserve(number_output_neurons);
    for (unsigned neuron_number = 0; neuron_number < number_output_neurons; ++neuron_number)
    {
      auto neuron = Neuron(
        neuron_number,
        dropout_rate <= 0.0 ? Neuron::Type::Normal : Neuron::Type::Dropout,
        dropout_rate);
      neurons.emplace_back(neuron);
    }
    return neurons;
  }

private:
  inline void apply_none_update(WeightParam& weight_param, double raw_gradient, double learning_rate) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    double new_weight = weight_param.get_value() - learning_rate * raw_gradient;
    weight_param.set_raw_gradient(raw_gradient);
    weight_param.set_value(new_weight);
  }

  inline void apply_sgd_update(WeightParam& weight_param, double raw_gradient, double learning_rate, double momentum, bool is_bias) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    
    if (!is_bias && weight_param.get_weight_decay() > 0.0)
    {
      raw_gradient += weight_param.get_weight_decay() * weight_param.get_value();
    }

    double previous_velocity = weight_param.get_velocity();
    double velocity = momentum * previous_velocity + raw_gradient;

    double new_weight = weight_param.get_value() - learning_rate * velocity;

    weight_param.set_raw_gradient(raw_gradient);
    weight_param.set_value(new_weight);
    weight_param.set_velocity(velocity);
  }

  inline void apply_adam_update(WeightParam& weight_param, double raw_gradient, double learning_rate, double beta1, double beta2, double epsilon, bool is_bias) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    // Update timestep
    weight_param.increment_timestep();
    const auto& time_step = weight_param.get_timestep();

    // Update moments
    double m = beta1 * weight_param.get_first_moment_estimate() + (1.0 - beta1) * raw_gradient;
    double v = beta2 * weight_param.get_second_moment_estimate() + (1.0 - beta2) * raw_gradient * raw_gradient;

    weight_param.set_first_moment_estimate(m);
    weight_param.set_second_moment_estimate(v);

    double m_hat = m / (1.0 - std::pow(beta1, time_step));
    double v_hat = v / (1.0 - std::pow(beta2, time_step));

    double adam_update = learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);

    // Apply decoupled weight decay (skip if bias)
    double decayed_weight = is_bias ? weight_param.get_value()
      : weight_param.get_value() * (1.0 - learning_rate * weight_param.get_weight_decay());

    double new_weight = decayed_weight - adam_update;

    weight_param.set_value(new_weight);
    weight_param.set_raw_gradient(raw_gradient);
  }

  inline void apply_adamw_update(
    WeightParam& weight_param,
    double raw_gradient,           // unclipped, averaged over batch
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon
  ) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    // Update timestep
    weight_param.increment_timestep();
    const auto& time_step = weight_param.get_timestep();

    // Update biased first and second moment estimates
    weight_param.set_first_moment_estimate(beta1 * weight_param.get_first_moment_estimate() + (1.0 - beta1) * raw_gradient);
    weight_param.set_second_moment_estimate(beta2 * weight_param.get_second_moment_estimate() + (1.0 - beta2) * (raw_gradient * raw_gradient));

    // Compute bias-corrected moments
    double first_moment_estimate = weight_param.get_first_moment_estimate();
    double m_hat = first_moment_estimate / (1.0 - std::pow(beta1, time_step));

    auto second_moment_estimate = weight_param.get_second_moment_estimate();
    double v_hat = second_moment_estimate / (1.0 - std::pow(beta2, time_step));

    // AdamW update rule
    double weight_update = learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon));

    // Decoupled weight decay
    auto new_weight = weight_param.get_value();
    new_weight *= (1.0 - learning_rate * weight_param.get_weight_decay());

    // Apply update
    new_weight -= weight_update;

    weight_param.set_value(new_weight);
    weight_param.set_raw_gradient(raw_gradient);
  }

  inline void apply_nadam_update(
    WeightParam& weight_param,
    double raw_gradient,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon
  ) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    // Update timestep
    weight_param.increment_timestep();
    const auto& time_step = weight_param.get_timestep();

    // These moment estimate updates are identical to Adam
    weight_param.set_first_moment_estimate(beta1 * weight_param.get_first_moment_estimate() + (1.0 - beta1) * raw_gradient);
    weight_param.set_second_moment_estimate(beta2 * weight_param.get_second_moment_estimate() + (1.0 - beta2) * (raw_gradient * raw_gradient));

    double m_hat = weight_param.get_first_moment_estimate() / (1.0 - std::pow(beta1, time_step));
    double v_hat = weight_param.get_second_moment_estimate() / (1.0 - std::pow(beta2, time_step));

    // Nadam's key difference:
    // It combines the momentum from the historical gradient (m_hat) with the
    // momentum from the current gradient.
    double corrected_gradient = (beta1 * m_hat) + ((1.0 - beta1) * raw_gradient) / (1.0 - std::pow(beta1, time_step));

    // The denominator is the same as Adam's
    double weight_update = learning_rate * (corrected_gradient / (std::sqrt(v_hat) + epsilon));

    // Apply the final update (No decoupled weight decay)
    double new_weight = weight_param.get_value() - weight_update;

    weight_param.set_value(new_weight);
    weight_param.set_raw_gradient(raw_gradient);
  }

  inline void apply_nadamw_update(
    WeightParam& weight_param,
    double raw_gradient,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    bool is_bias
  ) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    // 1. Increment timestep
    weight_param.increment_timestep();
    const long long time_step = weight_param.get_timestep();

    // 2. Update biased moment estimates
    double first_moment = beta1 * weight_param.get_first_moment_estimate()
      + (1.0 - beta1) * raw_gradient;
    double second_moment = beta2 * weight_param.get_second_moment_estimate()
      + (1.0 - beta2) * (raw_gradient * raw_gradient);

    weight_param.set_first_moment_estimate(first_moment);
    weight_param.set_second_moment_estimate(second_moment);

    // 3. Bias corrections
    double m_hat = first_moment / (1.0 - std::pow(beta1, time_step));
    double v_hat = second_moment / (1.0 - std::pow(beta2, time_step));

    // 4. NAdam momentum blend (with bias correction for gradient term)
    double m_nadam = beta1 * m_hat
      + ((1.0 - beta1) * raw_gradient) / (1.0 - std::pow(beta1, time_step));

    // 5. Adaptive step
    double step = m_nadam / (std::sqrt(v_hat) + epsilon);

    // 6. Decoupled weight decay
    double w = weight_param.get_value();
    if (!is_bias)
    {
      w *= (1.0 - learning_rate * weight_param.get_weight_decay());
    }

    // 7. Apply update
    w -= learning_rate * step;
    weight_param.set_value(w);
    weight_param.set_raw_gradient(raw_gradient);
  }

  unsigned _layer_index;
  LayerType _layer_type;
  activation _activation;
  OptimiserType _optimiser_type;
  int _residual_layer_number;

  unsigned _number_input_neurons;  //  number of neurons in previous layer
  unsigned _number_output_neurons; //  number of neurons in this layer

  // N_prev = number of neurons in previous layer
  // N_this = number of neurons in this layer
  // Size: [N_prev][N_this]
  std::vector<std::vector<WeightParam>> _weights;
  std::vector<WeightParam> _bias_weights;

  std::vector<Neuron> _neurons;
  
  ResidualProjector* _residual_projector;
};