#pragma once
#include <memory>
#include <shared_mutex>

#include "gradientsandoutputs.h"
#include "layer.h"
#include "layerdetails.h"
#include "logger.h"
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
    const std::vector<std::unique_ptr<Layer>>& layers
    ) noexcept;
  
  virtual ~Layers();

  const std::vector<std::unique_ptr<Layer>>& get_layers() const;
  std::vector<std::unique_ptr<Layer>>& get_layers();

  const Layer& operator[](unsigned index) const;
  Layer& operator[](unsigned index);

  int get_residual_layer_number(unsigned index) const noexcept;
  const ResidualProjector* get_residual_layer_projector(unsigned index) const noexcept;

  [[nodiscard]] inline size_t size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return _layers.size();
  }

  [[nodiscard]] inline const Layer& input_layer() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return *_layers.front();
  }

  [[nodiscard]] inline const Layer& hidden_layer(unsigned index) const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
#if VALIDATE_DATA == 1
    if (index == 0)
    {
      Logger::panic("Trying to get hidden layer information that is actually the input layer!");
    }
    if (index >= static_cast<unsigned>(_layers.size() -1))
    {
      Logger::panic("Trying to get hidden layer information past the number of hidden layers!");
    }
#endif
    return *_layers.at(index);
  }

  [[nodiscard]] inline const Layer& output_layer() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return *_layers.back();
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
  [[nodiscard]] inline const Layer& layer(unsigned index) const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
#if VALIDATE_DATA == 1
    if (index >= static_cast<unsigned>(_layers.size()))
    {
      Logger::panic("Trying to get layer information past the number of layers!");
    }
#endif
    return *_layers.at(index);
  }

  void calculate_forward_feed(
    const NeuralNetworkOptions& options,
    std::vector<GradientsAndOutputs>& gradients_and_output,
    std::vector<std::vector<double>>::const_iterator inputs_begin,
    size_t batch_size,
    std::vector<HiddenStates>& hidden_states,
    bool is_training) const;

  void calculate_back_propagation(
    const NeuralNetworkOptions& options,
    std::vector<GradientsAndOutputs>& gradients,
    std::vector<std::vector<double>>::const_iterator outputs_begin,
    size_t batch_size,
    const std::vector<HiddenStates>& hidden_states) const;

  void calculate_back_propagation_input_layer(
    const NeuralNetworkOptions& options,
    std::vector<GradientsAndOutputs>& gradients,
    size_t batch_size) const;

  void calculate_back_propagation_output_layer(
    const NeuralNetworkOptions& options,
    std::vector<GradientsAndOutputs>& gradients,
    std::vector<std::vector<double>>::const_iterator outputs_begin,
    size_t batch_size,
    const std::vector<HiddenStates>& hidden_states) const;

  void calculate_back_propagation_hidden_layers(
    const NeuralNetworkOptions& options,
    std::vector<GradientsAndOutputs>& gradients,
    size_t batch_size,
    const std::vector<HiddenStates>& hidden_states) const;

  void update_weights(
    const NeuralNetworkOptions& options,
    const std::vector<GradientsAndOutputs>& batch_gradients,
    double learning_rate,
    size_t batch_size,
    const std::vector<HiddenStates>& hidden_states);

  ResidualProjector* create_residual_projector(const activation& activation_method, int residual_layer_number, int number_of_neurons_in_current_layer, double weight_decay);
  static std::unique_ptr<Layer> create_input_layer(unsigned num_neurons_in_this_layer, int residual_layer_number, int number_of_threads, bool has_bias);
  std::unique_ptr<Layer> create_hidden_layer(double weight_decay, const Layer& previous_layer, const OptimiserType& optimiser_type, int residual_layer_number, double dropout_rate, const LayerDetails& layer_details, int number_of_threads, bool has_bias, double momentum);
  std::unique_ptr<Layer> create_output_layer(unsigned num_neurons_in_this_layer, const Layer& previous_layer, const std::vector<OutputLayerDetails>& output_layer_details, int number_of_threads, bool has_bias);
  std::unique_ptr<Layer> create_branched_output_layer(unsigned num_neurons_in_this_layer, const Layer& previous_layer, const std::vector<LayerDetails::BranchDetails>& branched_outputs, int number_of_threads, bool has_bias);

  int compute_residual_layer(int current_layer_index, int residual_layer_jump) const;

  std::vector<std::unique_ptr<Layer>> _layers;

  std::vector<GradientsAndOutputs> _training_gradients_buffer;
  std::vector<HiddenStates> _training_hidden_states_buffer;

  mutable std::shared_mutex _mutex;
  TaskQueuePool<void>* _update_weights_pool;
};