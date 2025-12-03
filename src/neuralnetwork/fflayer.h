#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif

#include "activation.h"
#include "errorcalculation.h"
#include "neuron.h"
#include "optimiser.h"
#include "weightparam.h"

#include <cassert>
#include <vector>

class Neuron;
class FFLayer
{
protected:
  friend class Layers;
public:
  enum class LayerType
  {
    Input,
    Hidden,
    Output
  };
private:
  FFLayer(unsigned layer_index,
    unsigned num_neurons_in_previous_layer, 
    unsigned num_neurons_in_this_layer, 
    unsigned num_neurons_in_next_layer, 
    double weight_decay,
    LayerType layer_type, 
    const activation::method& activation_method, 
    const OptimiserType& optimiser_type, 
    double dropout_rate);

public:
  FFLayer(
    unsigned layer_index,
    const std::vector<Neuron>& neurons,
    unsigned number_input_neurons,
    LayerType layer_type,
    OptimiserType optimiser_type,
    const activation::method& activation_method,
    const std::vector<std::vector<WeightParam>>& weights,
    const std::vector<WeightParam>& bias_weights
    );

  FFLayer(const FFLayer& src) noexcept;
  FFLayer(FFLayer&& src) noexcept;
  FFLayer& operator=(const FFLayer& src) noexcept;
  FFLayer& operator=(FFLayer&& src) noexcept;
  virtual ~FFLayer();

  unsigned number_neurons() const noexcept;
  unsigned number_neurons_with_bias() const noexcept;
  const std::vector<Neuron>& get_neurons() const noexcept;
  std::vector<Neuron>& get_neurons() noexcept;

  const Neuron& get_neuron(unsigned index) const;
  Neuron& get_neuron(unsigned index);

  LayerType layer_type() const { 
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    return _layer_type; 
  }

public:
  static FFLayer create_input_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, double weight_decay );
  static FFLayer create_hidden_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, double weight_decay, const FFLayer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, double dropout_rate);
  static FFLayer create_output_layer(unsigned num_neurons_in_this_layer, double weight_decay, const FFLayer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type);

  std::vector<std::vector<double>> calculate_forward_feed(
    const FFLayer& previous_layer,
    const std::vector<std::vector<double>>& previous_layer_inputs,
    std::vector<std::vector<double>>& pre_activation_sums,
    bool is_training) const;

  std::vector<std::vector<double>> calculate_output_gradients(
    const std::vector<std::vector<double>>& target_outputs,
    const std::vector<std::vector<double>>& given_outputs,
    const std::vector<std::vector<double>>& pre_activation_sums,
    double gradient_clip_threshold,
    ErrorCalculation::type error_calculation_type) const;

  void calculate_error_deltas(
    std::vector<std::vector<double>>& deltas,
    const std::vector<std::vector<double>>& target_outputs,
    const std::vector<std::vector<double>>& given_outputs,
    ErrorCalculation::type error_calculation_type) const;

  void calculate_mse_error_deltas(
    std::vector<std::vector<double>>& deltas,
    const std::vector<std::vector<double>>& target_outputs,
    const std::vector<std::vector<double>>& given_outputs) const;

  void calculate_bce_error_deltas(
    std::vector<std::vector<double>>& deltas,
    const std::vector<std::vector<double>>& target_outputs,
    const std::vector<std::vector<double>>& given_outputs) const;

  std::vector<std::vector<double>> calculate_hidden_gradients(
    const FFLayer& next_layer,
    const std::vector<std::vector<double>>& next_grad_matrix,
    const std::vector<std::vector<double>>& output_matrix,
    const std::vector<std::vector<double>>& pre_activation_sums,
    double gradient_clip_threshold) const;

  inline unsigned number_input_neurons(bool add_bias) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    return _number_input_neurons + (add_bias ? 1 : 0);
  }

  inline unsigned get_layer_index() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    return _layer_index;
  }

  void apply_weight_gradient(const double gradient, const double learning_rate, bool is_bias, WeightParam& weight_param, double clipping_scale, double gradient_clip_threshold);

  // optimisers
  // TODO THOSE SHOULD ALL BE PRIVATE
  static double clip_gradient(double gradient, double gradient_clip_threshold);
  // Removed static apply_*_update methods

  const std::vector<std::vector<WeightParam>>& get_weight_params() const
  {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    return _weights;
  }

  const WeightParam& get_weight_param(unsigned input_neuron_number, unsigned neuron_index) const
  {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    // TODO Valiate
    return _weights[input_neuron_number][neuron_index];
  }
  WeightParam& get_weight_param(unsigned input_neuron_number, unsigned neuron_index)
  {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    // TODO Valiate
    return _weights[input_neuron_number][neuron_index];
  }
  inline WeightParam& get_bias_weight_param(unsigned neuron_index) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    // TODO Valiate
    return _bias_weights[neuron_index];
  }

  bool has_bias() const noexcept;

  inline const std::vector<WeightParam>& get_bias_weight_params() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    // TODO Valiate and profile.
    return _bias_weights;
  }
  inline const OptimiserType get_optimiser_type() const noexcept {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    return _optimiser_type;
  }
  inline const activation& get_activation() const noexcept {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    return _activation;
  }
  inline const LayerType& get_layer_type() const noexcept {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    return _layer_type;
  }
  inline unsigned get_number_input_neurons() const {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    return _number_input_neurons;
  }
  inline unsigned get_number_output_neurons() const {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    return _number_output_neurons;
  }

private:
  void clean();
  void resize_weights(
    const activation& activation_method,
    unsigned number_input_neurons, 
    unsigned number_output_neurons, 
    double weight_decay);

  unsigned _layer_index;
  std::vector<Neuron> _neurons;
  unsigned _number_input_neurons;  //  number of neurons in previous layer
  unsigned _number_output_neurons; //  number of neurons in this layer
  LayerType _layer_type;
  OptimiserType _optimiser_type;
  activation _activation;

  // N_prev = number of neurons in previous layer
  // N_this = number of neurons in this layer
  // Size: [N_prev][N_this]
  std::vector<std::vector<WeightParam>> _weights;
  std::vector<WeightParam> _bias_weights;
};
