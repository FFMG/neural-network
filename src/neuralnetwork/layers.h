#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif

#include <memory>
#include <shared_mutex>

#include "gradientsandoutputs.h"
#include "layer.h"
#include "layerdetails.h"
#include "neuralnetworkoptions.h"
#include "optimiser.h"
#include "residualprojector.h"
#include "taskqueue.h"

class Layers
{
public:
  Layers(const NeuralNetworkOptions& options) noexcept;
  Layers(const Layers& layers) noexcept;
  Layers(Layers&& layers) noexcept;
  
  Layers& operator=(const Layers& layers) noexcept;
  Layers& operator=(Layers&& layers) noexcept;

  Layers(
    const NeuralNetworkOptions& options,
    const std::vector<std::unique_ptr<Layer>>& layers,
    double weight_decay
    ) noexcept;
  
  virtual ~Layers();

  const std::vector<std::unique_ptr<Layer>>& get_layers() const;
  std::vector<std::unique_ptr<Layer>>& get_layers();

  const Layer& operator[](unsigned index) const;
  Layer& operator[](unsigned index);

  int get_residual_layer_number(unsigned index) const noexcept;
  const ResidualProjector* get_residual_layer_projector(unsigned index) const noexcept;

  inline size_t size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return _layers.size();
  }

  inline const Layer& input_layer() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return *_layers.front();
  }

  inline const Layer& hidden_layer(unsigned index) const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return *_layers.at(index);
  }

  inline const Layer& output_layer() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return *_layers.back();
  }

  inline double get_weight_decay() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return _weight_decay;
  }

  void train(
    const NeuralNetworkOptions& options, 
    const double learning_rate,
    std::vector<std::vector<double>>::const_iterator& training_inputs, 
    std::vector<std::vector<double>>::const_iterator& training_outputs,
    const size_t batch_size);
  std::vector<std::vector<double>> think(const NeuralNetworkOptions& options, const std::vector<std::vector<double>>& inputs) const;
  std::vector<double> think(const NeuralNetworkOptions& options, const std::vector<double>& inputs) const;

private:
  const std::vector<HiddenStates> calculate_forward_feed(
    const NeuralNetworkOptions& options,
    std::vector<GradientsAndOutputs>& gradients_and_output,
    std::vector<std::vector<double>>::const_iterator inputs_begin,
    size_t batch_size,
    bool is_training) const;

  void calculate_back_propagation(
    const NeuralNetworkOptions& options,
    std::vector<GradientsAndOutputs>& gradients,
    std::vector<std::vector<double>>::const_iterator outputs_begin,
    size_t batch_size,
    const std::vector<HiddenStates>& hidden_states) const;

  void update_weights(
    const NeuralNetworkOptions& options,
    const std::vector<GradientsAndOutputs>& batch_gradients,
    double learning_rate,
    const std::vector<HiddenStates>& hidden_states);

  void calculate_back_propagation_input_layer(
    const NeuralNetworkOptions& options,
    std::vector<GradientsAndOutputs>& gradients) const;

  void calculate_back_propagation_output_layer(
    const NeuralNetworkOptions& options,
    std::vector<GradientsAndOutputs>& gradients,
    std::vector<std::vector<double>>::const_iterator outputs_begin,
    size_t batch_size,
    const std::vector<HiddenStates>& hidden_states) const;

  void calculate_back_propagation_hidden_layers(
    const NeuralNetworkOptions& options,
    std::vector<GradientsAndOutputs>& gradients,
    const std::vector<HiddenStates>& hidden_states) const;

  ResidualProjector* create_residual_projector(const activation& activation_method, int residual_layer_number, int number_of_neurons_in_current_layer, double weight_decay);
  static std::unique_ptr<Layer> create_input_layer(unsigned num_neurons_in_this_layer, double weight_decay, int residual_layer_number, int number_of_threads);
  std::unique_ptr<Layer> create_hidden_layer(double weight_decay, const Layer& previous_layer, const OptimiserType& optimiser_type, int residual_layer_number, double dropout_rate, const LayerDetails& layer_details, int number_of_threads);
  std::unique_ptr<Layer> create_output_layer(unsigned num_neurons_in_this_layer, double weight_decay, const Layer& previous_layer, const activation& activation, const OptimiserType& optimiser_type, int residual_layer_number, int number_of_threads);

  int compute_residual_layer(int current_layer_index, int residual_layer_jump) const;

  std::vector<std::unique_ptr<Layer>> _layers;
  double _weight_decay;

  mutable std::shared_mutex _mutex;
  TaskQueuePool<void>* _update_weights_pool;
};