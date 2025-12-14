#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif

#include "layer.h"
#include "errorcalculation.h"
#include "gradientsandoutputs.h"
#include "hiddenstate.h"
#include "neuron.h"
#include "optimiser.h"
#include "weightparam.h"

#include <cassert>
#include <vector>

class Neuron;
class ElmanRNNLayer final : public Layer
{
protected:
  friend class Layers;

public:
  ElmanRNNLayer(unsigned layer_index,
    unsigned num_neurons_in_previous_layer, 
    unsigned num_neurons_in_this_layer, 
    unsigned num_neurons_in_next_layer, 
    double weight_decay,
    LayerType layer_type, 
    const activation::method& activation_method, 
    const OptimiserType& optimiser_type, 
    int residual_layer_number,
    double dropout_rate);

public:
  ElmanRNNLayer(
    unsigned layer_index,
    const std::vector<Neuron>& neurons,
    unsigned number_input_neurons,
    LayerType layer_type,
    OptimiserType optimiser_type,
    int residual_layer_number,
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

  const std::vector<Neuron>& get_neurons() const noexcept;
  std::vector<Neuron>& get_neurons() noexcept;

  const Neuron& get_neuron(unsigned index) const;
  Neuron& get_neuron(unsigned index);

  int residual_layer_number() const override;

public:
  std::vector<double> calculate_forward_feed(
      GradientsAndOutputs& gradients_and_outputs,
      const Layer&previous_layer,
      const std::vector<double> &previous_layer_inputs,
      const std::vector<double> &residual_output_values,
      std::vector<HiddenState> &hidden_states,
      bool is_training) const override;

  void calculate_output_gradients(
      GradientsAndOutputs& gradients_and_outputs,
      const std::vector<double> &target_outputs,
      const std::vector<HiddenState> &hidden_states,
      double gradient_clip_threshold,
      ErrorCalculation::type error_calculation_type) const  override;

  void calculate_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    ErrorCalculation::type error_calculation_type) const;

  void calculate_mse_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs) const;

  void calculate_bce_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs) const;

  void calculate_hidden_gradients(
      GradientsAndOutputs& gradients_and_outputs,
      const Layer&next_layer,
      const std::vector<double> &next_grad_matrix,
      const std::vector<double> &output_matrix,
      const std::vector<HiddenState> &hidden_states,
      double gradient_clip_threshold) const override;

  void apply_weight_gradient(const double gradient, const double learning_rate, bool is_bias, WeightParam& weight_param, double clipping_scale, double gradient_clip_threshold) override;

  static double clip_gradient(double gradient, double gradient_clip_threshold);

  const std::vector<std::vector<WeightParam>>& get_weight_params() const override;
  const std::vector<std::vector<WeightParam>>& get_recurrent_weight_params() const;
  std::vector<std::vector<WeightParam>>& get_recurrent_weight_params();

  const WeightParam& get_weight_param(unsigned input_neuron_number, unsigned neuron_index) const override;
  WeightParam& get_weight_param(unsigned input_neuron_number, unsigned neuron_index) override;
  
  WeightParam& get_bias_weight_param(unsigned neuron_index) override;

  bool has_bias() const noexcept override;

  const std::vector<WeightParam>& get_bias_weight_params() const override;

  const std::vector<std::vector<WeightParam>>& get_residual_weight_params() const override;
  std::vector<std::vector<WeightParam>>& get_residual_weight_params() override;
  std::vector<WeightParam>& get_residual_weight_params(unsigned neuron_index) override;

  const OptimiserType get_optimiser_type() const noexcept;

  Layer* clone() const override;

private:
  void resize_weights(
    unsigned number_input_neurons, 
    unsigned number_output_neurons, 
    double weight_decay);

  std::vector<Neuron> _neurons;
  OptimiserType _optimiser_type;

  // N_prev = number of neurons in previous layer
  // N_this = number of neurons in this layer
  // Size: [N_prev][N_this]
  std::vector<std::vector<WeightParam>> _weights;
  // Size: [N_this][N_this]
  std::vector<std::vector<WeightParam>> _recurrent_weights;
  std::vector<WeightParam> _bias_weights;
  std::vector<std::vector<WeightParam>> _residual_weights;
  int _residual_layer_number;
};
