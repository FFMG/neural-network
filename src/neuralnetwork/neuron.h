#pragma once
#include "activation.h"
#include "layer.h"
#include "logger.h"
#include "optimiser.h"
#include "weightparam.h"
#include "./libraries/instrumentor.h"

#include <vector>

#define LEARNING_ALPHA double(0.5)    // momentum, multiplier of last gradient, [0.0..1.0]

class Layer;
class Neuron
{
public:
  enum class Type
  {
    Normal,
    Bias
  };

public:
  Neuron(
    unsigned index, 
    const activation& activation,
    const std::vector<WeightParam>& weight_params,
    const OptimiserType& optimiser_type,
    const Type& type,
    const Logger& logger
    );
    
  Neuron(
    unsigned num_neurons_prev_layer,
    unsigned num_neurons_current_layer,
    unsigned num_neurons_next_layer,
    unsigned index, 
    const activation& activation,
    const OptimiserType& optimiser_type,
    const Type& type,
    const Logger& logger
    );

  Neuron(const Neuron& src) noexcept;
  Neuron& operator=(const Neuron& src) noexcept;
  Neuron(Neuron&& src) noexcept;
  Neuron& operator=(Neuron&& src) noexcept;

  virtual ~Neuron();
  
  double calculate_forward_feed(const Layer&, 
    const std::vector<double>& previous_layer_output_values,
    const std::vector<double>& residual_output_values
  ) const;
  
  double calculate_output_gradients(double target_value, double output_value) const;
  
  double calculate_hidden_gradients(const Layer& next_layer, const std::vector<double>& activation_gradients, double output_value) const;

  void apply_weight_gradients(Layer& previous_layer, const std::vector<double>& gradients, const double learning_rate, unsigned epoch);
  void apply_residual_projection_gradients(Layer& layer, const std::vector<double>& residual_outputs, const std::vector<double>& gradients, double learning_rate);

  unsigned get_index() const;
  const std::vector<WeightParam>& get_weight_params() const;

  const OptimiserType& get_optimiser_type() const;
  bool is_bias() const;

private:
  void Clean();
  double sum_of_derivatives_of_weights(const Layer& next_layer, const std::vector<double>& activation_gradients) const;
  double get_output_weight(int index) const;

  void apply_weight_gradient(const double gradient, const double learning_rate, bool is_bias, WeightParam& weight_param);

  // optimisers
  void apply_none_update(WeightParam& weight_param, double raw_gradient, double learning_rate) const;
  void apply_sgd_update(WeightParam& weight_param, double raw_gradient, double learning_rate, 
    double momentum, bool is_bias) const;
  void apply_adam_update(WeightParam& weight_param, double raw_gradient, double learning_rate, 
    double beta1,
    double beta2,
    double epsilon,
    bool is_bias) const;
  void apply_adamw_update(
    WeightParam& weight_param,
    double raw_gradient,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon
  ) const;
  void apply_nadam_update(
      WeightParam& weight_param,
      double raw_gradient,
      double learning_rate,
      double beta1,
      double beta2,
      double epsilon
  ) const;
  void apply_nadamw_update(
      WeightParam& weight_param,
      double raw_gradient,
      double learning_rate,
      double beta1,
      double beta2,
      double epsilon,
      bool is_bias
  ) const;  
  
  double clip_gradient(double gradient) const;
  
  // data to save...
  unsigned _index;
  activation _activation_method;
  std::vector<WeightParam> _weight_params;
  OptimiserType _optimiser_type;
  
  const double _alpha; // [0.0..n] multiplier of last weight change (momentum)
  Type _type;
  Logger _logger;
};