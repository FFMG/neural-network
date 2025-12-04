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
#include "baselayer.h"
#include "hiddenstate.h"

#include <cassert>
#include <vector>

class Neuron;
class ElmanRNNLayer final : public BaseLayer
{
protected:
  friend class Layers;

private:
  ElmanRNNLayer(unsigned layer_index,
    unsigned num_neurons_in_previous_layer, 
    unsigned num_neurons_in_this_layer, 
    unsigned num_neurons_in_next_layer, 
    double weight_decay,
    LayerType layer_type, 
    const activation::method& activation_method, 
    const OptimiserType& optimiser_type, 
    double dropout_rate);

public:
  ElmanRNNLayer(
    unsigned layer_index,
    const std::vector<Neuron>& neurons,
    unsigned number_input_neurons,
    LayerType layer_type,
    OptimiserType optimiser_type,
    const activation::method& activation_method,
    const std::vector<std::vector<WeightParam>>& weights,
    const std::vector<std::vector<WeightParam>>& recurrent_weights,
    const std::vector<WeightParam>& bias_weights
    );

  ElmanRNNLayer(const ElmanRNNLayer& src) noexcept;
  ElmanRNNLayer(ElmanRNNLayer&& src) noexcept;
  ElmanRNNLayer& operator=(const ElmanRNNLayer& src) noexcept;
  ElmanRNNLayer& operator=(ElmanRNNLayer&& src) noexcept;
  virtual ~ElmanRNNLayer();

  unsigned number_neurons() const noexcept override;
  const std::vector<Neuron>& get_neurons() const noexcept;
  std::vector<Neuron>& get_neurons() noexcept;

  const Neuron& get_neuron(unsigned index) const;
  Neuron& get_neuron(unsigned index);

  LayerType layer_type() const override;

public:
  static ElmanRNNLayer create_input_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, double weight_decay );
  static ElmanRNNLayer create_hidden_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, double weight_decay, const ElmanRNNLayer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, double dropout_rate);
  static ElmanRNNLayer create_output_layer(unsigned num_neurons_in_this_layer, double weight_decay, const ElmanRNNLayer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type);

  std::vector<std::vector<double>> calculate_forward_feed(
      const BaseLayer &previous_layer,
      const std::vector<std::vector<double>> &previous_layer_inputs,
      const std::vector<std::vector<double>> &residual_output_values,
      std::vector<std::vector<HiddenState>> &hidden_states,
      bool is_training) const override;

  std::vector<std::vector<double>> calculate_output_gradients(
      const std::vector<std::vector<double>> &target_outputs,
      const std::vector<std::vector<double>> &given_outputs,
      const std::vector<std::vector<HiddenState>> &hidden_states,
      double gradient_clip_threshold,
      ErrorCalculation::type error_calculation_type) const override;

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
      const BaseLayer &next_layer,
      const std::vector<std::vector<double>> &next_grad_matrix,
      const std::vector<std::vector<double>> &output_matrix,
      const std::vector<std::vector<HiddenState>> &hidden_states,
      double gradient_clip_threshold) const override;

  unsigned number_input_neurons(bool add_bias) const noexcept override;

  unsigned get_layer_index() const noexcept override;

  void apply_weight_gradient(const double gradient, const double learning_rate, bool is_bias, WeightParam& weight_param, double clipping_scale, double gradient_clip_threshold);

  static double clip_gradient(double gradient, double gradient_clip_threshold);

  const std::vector<std::vector<WeightParam>>& get_weight_params() const override;
  const std::vector<std::vector<WeightParam>>& get_recurrent_weight_params() const;

  const WeightParam& get_weight_param(unsigned input_neuron_number, unsigned neuron_index) const override;
  WeightParam& get_weight_param(unsigned input_neuron_number, unsigned neuron_index) override;
  
  WeightParam& get_bias_weight_param(unsigned neuron_index);

  bool has_bias() const noexcept override;

  const std::vector<WeightParam>& get_bias_weight_params() const override;

  const OptimiserType get_optimiser_type() const noexcept;
  
  const activation& get_activation() const noexcept override;
  
  unsigned get_number_output_neurons() const;

private:
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
  // Size: [N_this][N_this]
  std::vector<std::vector<WeightParam>> _recurrent_weights;
  std::vector<WeightParam> _bias_weights;
};
