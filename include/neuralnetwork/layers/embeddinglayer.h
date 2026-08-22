#pragma once
#include "layer.h"

#include <vector>


namespace myoddweb::nn
{
// Categorical Entity Embedding Layer: maps discrete integer indices (e.g.
// asset ticker ID, market regime, day of week, discrete event flags) into
// continuous dense learned vector representations of fixed dimension D.
// Given K input categorical features and embedding dimension D, produces an
// output vector of size K * D (the concatenation of the K embedding vectors).
// Learns embedding weights via standard backpropagation and configured optimisers.
class EmbeddingLayer final : public Layer
{
public:
  EmbeddingLayer(
    unsigned layer_index,
    unsigned num_neurons_in_previous_layer,
    unsigned vocabulary_size,
    unsigned embedding_dimension,
    double weight_decay,
    const Role layer_role,
    const activation& activation_method,
    const OptimiserType& optimiser_type,
    int residual_layer_number,
    double dropout_rate,
    ResidualProjector* residual_projector,
    int number_of_threads,
    double momentum,
    std::optional<uint32_t> seed);

  EmbeddingLayer(
    unsigned layer_index,
    const Role layer_role,
    const OptimiserType optimiser_type,
    int residual_layer_number,
    unsigned vocabulary_size,
    unsigned embedding_dimension,
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
    const ResidualProjector* residual_projector,
    int number_of_threads,
    const layer_activation_helper& lah,
    double momentum);

  EmbeddingLayer(const EmbeddingLayer& src) noexcept;
  EmbeddingLayer(EmbeddingLayer&& src) noexcept;
  EmbeddingLayer& operator=(const EmbeddingLayer& src) noexcept;
  EmbeddingLayer& operator=(EmbeddingLayer&& src) noexcept;
  virtual ~EmbeddingLayer() override;

  [[nodiscard]] Architecture get_layer_architecture() const noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
    return Architecture::Embedding;
  }

  [[nodiscard]] inline unsigned get_vocabulary_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
    return _vocabulary_size;
  }

  [[nodiscard]] inline unsigned get_embedding_dimension() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
    return _embedding_dimension;
  }

  void calculate_forward_feed(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& previous_layer,
    const std::vector<std::vector<double>>& batch_residual_output_values,
    std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    bool is_training) const override;

  void calculate_output_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    std::vector<std::vector<double>>::const_iterator target_outputs_begin,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size) const override;

  void calculate_hidden_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& next_layer,
    const std::vector<std::vector<double>>& batch_next_grad_matrix,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    int bptt_max_ticks) const override;

  void calculate_hidden_gradients_from_output_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<std::vector<double>>& batch_output_gradients,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    int bptt_max_ticks) const override;

  void calculate_and_store_gradients(
    const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& hidden_states,
    const Layer& previous_layer,
    size_t batch_size,
    int bptt_max_ticks) override;

  double get_gradient_norm_sq() const override;

  void accumulate_swa_average_impl(const Layer& snapshot, size_t existing_swa_count) override;

  void apply_stored_gradients(double learning_rate, double clipping_scale) override;

  void zero_gradients() override;

  Layer* clone() const override;

private:
  void calculate_and_store_gradients_chunk(
    size_t start,
    size_t end,
    const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    unsigned prev_layer_index,
    unsigned this_layer_index,
    unsigned num_inputs,
    unsigned num_outputs,
    size_t num_time_steps,
    std::vector<double>& w_grads_out) const;

  void run_post_gemm_backward(
    size_t start,
    size_t end,
    size_t N_this,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& batch_hidden_states,
    const std::vector<double>& flattened_this_grads_buffer) const;

  unsigned _vocabulary_size;
  unsigned _embedding_dimension;
  mutable std::vector<std::vector<double>> _thread_w_grads;
};

} // namespace myoddweb::nn
