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
#include "outputlayerdetails.h"

#include <vector>
#include <cstdint>

class FFOutputLayer final : public FFLayer
{
protected:
  friend class Layers;

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

  bool has_bias() const noexcept override;
  
  Layer* clone() const override;

  inline const activation& get_activation() const noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
    Logger::panic("Trying to get an activation layer without passing an neuron index!");
  }

protected:
  virtual void run_post_gemm(
    size_t start,
    size_t end,
    size_t N_this,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<std::vector<double>>& batch_residual_output_values,
    std::vector<HiddenStates>& batch_hidden_states,
    bool is_training) const;

private:
  void run_output_gradients(
    size_t start,
    size_t end,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    std::vector<std::vector<double>>::const_iterator target_outputs_begin,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t num_neurons) const;

  std::vector<OutputLayerDetails> _output_layer_details;

  [[nodiscard]] inline const activation& get_activation(unsigned neuron_index) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
#if VALIDATE_DATA == 1
    if (neuron_index >= _activations.size())
    {
      Logger::panic("Trying to get an activation layer outside of the index!");
    }
#endif
    return _activations[neuron_index];
  }

  [[nodiscard]] inline bool get_is_not_using_activation_derivatives(unsigned neuron_index) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
#if VALIDATE_DATA == 1
    if (neuron_index >= _is_not_using_activation_derivatives.size())
    {
      Logger::panic("Trying to get if using an activation derivative outside of the index!");
    }
#endif
    return _is_not_using_activation_derivatives[neuron_index] != 0;
  }

  void create_activation_per_neuron(const std::vector<OutputLayerDetails>& output_layer_details);
  void create_using_activation_derivatives_per_neuron(const std::vector<OutputLayerDetails>& output_layer_details);

  void calculate_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs) const;

  std::vector<activation> _activations;
  std::vector<uint8_t> _is_not_using_activation_derivatives;
};