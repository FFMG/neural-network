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
#include "outputlayer.h"

#include <vector>
#include <cstdint>

class FFOutputLayer final : public FFLayer, OutputLayer
{
public:
  FFOutputLayer(
    unsigned layer_index,
    const std::vector<OutputLayerDetails>& output_layer_details,
    unsigned num_neurons_in_previous_layer, 
    unsigned num_neurons_in_this_layer, 
    double weight_decay,
    const OptimiserType& optimiser_type, 
    int number_of_threads);

  FFOutputLayer(
    unsigned layer_index,
    const std::vector<OutputLayerDetails>& output_layer_details,
    const OptimiserType optimiser_type,
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
    size_t batch_size) const override;

  void calculate_hidden_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& next_layer,
    const std::vector<std::vector<double>>& batch_next_grad_matrix,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    int bptt_max_ticks) const override;

  void calculate_forward_feed(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& previous_layer,
    const std::vector<std::vector<double>>& batch_residual_output_values,
    std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    bool is_training) const override;
  
  Layer* clone() const override;

  inline const activation& get_activation() const noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
    Logger::panic("Trying to get an activation layer without passing an neuron index!");
  }

protected:
  virtual void run_post_gemm(
    const size_t start,
    const size_t end,
    const size_t N_this,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<std::vector<double>>& batch_residual_output_values,
    std::vector<HiddenStates>& batch_hidden_states,
    bool is_training) const;

private:
  void run_output_gradients(
    const size_t start,
    const size_t end,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    std::vector<std::vector<double>>::const_iterator target_outputs_begin,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t num_neurons) const;

  void calculate_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs) const;
};