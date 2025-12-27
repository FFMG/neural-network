#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif

#include "errorcalculation.h"
#include "hiddenstate.h"
#include "layer.h"
#include "taskqueue.h"

#include <vector>

class FFLayer final : public Layer
{
protected:
  friend class Layers;

public:
  FFLayer(unsigned layer_index,
    unsigned num_neurons_in_previous_layer, 
    unsigned num_neurons_in_this_layer, 
    double weight_decay,
    LayerType layer_type, 
    const activation::method& activation_method, 
    const OptimiserType& optimiser_type, 
    int residual_layer_number,
    double dropout_rate,
    ResidualProjector* residual_projector);

  FFLayer(const FFLayer& src) noexcept;
  FFLayer(FFLayer&& src) noexcept;
  FFLayer& operator=(const FFLayer& src) noexcept;
  FFLayer& operator=(FFLayer&& src) noexcept;
  virtual ~FFLayer();

public:
  void calculate_forward_feed(
      std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
      const Layer &previous_layer,
      const std::vector<std::vector<double>> &batch_residual_output_values,
      std::vector<HiddenStates> &batch_hidden_states,
      bool is_training) const override;

  void calculate_output_gradients(
      std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
      std::vector<std::vector<double>>::const_iterator target_outputs_begin,
      const std::vector<HiddenStates> &batch_hidden_states,
      ErrorCalculation::type error_calculation_type) const override;

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
      std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
      const Layer &next_layer,
      const std::vector<std::vector<double>> &batch_next_grad_matrix,
      const std::vector<HiddenStates> &batch_hidden_states,
      int bptt_max_ticks) const override;

  bool has_bias() const noexcept override;
  
  Layer* clone() const override;

  TaskQueuePool<void>* _task_queue_pool;
};