#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif

#include "errorcalculation.h"
#include "fflayer.h"
#include "layer.h"

#include <vector>

class FFOutputLayer final : public FFLayer
{
protected:
  friend class Layers;

public:
  FFOutputLayer(unsigned layer_index,
    unsigned num_neurons_in_previous_layer, 
    unsigned num_neurons_in_this_layer, 
    double weight_decay,
    const activation& activation_method, 
    const OptimiserType& optimiser_type, 
    int residual_layer_number,
    double dropout_rate,
    ResidualProjector* residual_projector,
    int number_of_threads);

  FFOutputLayer(
    unsigned layer_index,
    const activation activation,
    const OptimiserType optimiser_type,
    int residual_layer_number,
    unsigned number_input_neurons,
    unsigned number_output_neurons,
    const std::vector<Neuron>& neurons,
    const std::vector<double>& w_values,
    const std::vector<double>& w_grads,
    const std::vector<double>& w_velocities,
    const std::vector<double>& w_m1,
    const std::vector<double>& w_m2,
    const std::vector<long long>& w_timesteps,
    const std::vector<double>& w_decays,
    const std::vector<double>& b_values,
    const std::vector<double>& b_grads,
    const std::vector<double>& b_velocities,
    const std::vector<double>& b_m1,
    const std::vector<double>& b_m2,
    const std::vector<long long>& b_timesteps,
    const std::vector<double>& b_decays,
    const ResidualProjector* residual_projector,
    int number_of_threads
  ) noexcept;

  FFOutputLayer(const FFOutputLayer& src) noexcept;
  FFOutputLayer(FFOutputLayer&& src) noexcept;
  FFOutputLayer& operator=(const FFOutputLayer& src) noexcept;
  FFOutputLayer& operator=(FFOutputLayer&& src) noexcept;
  virtual ~FFOutputLayer();

public:
  void calculate_output_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    std::vector<std::vector<double>>::const_iterator target_outputs_begin,
    const std::vector<HiddenStates> &batch_hidden_states,
    const OutputLayerDetails& output_layer_detail) const override;

  bool has_bias() const noexcept override;
  
  Layer* clone() const override;
};