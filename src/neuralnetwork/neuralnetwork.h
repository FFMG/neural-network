#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif

#include <cassert>
#include <functional>
#include <map>
#include <shared_mutex>
#include <vector>

#include "./libraries/instrumentor.h"

#include "activation.h"
#include "adaptivelearningratescheduler.h"
#include "layers.h"
#include "neuron.h"
#include "optimiser.h"
#include "rng.h"
#include "taskqueue.h"
#include "neuralnetworkhelper.h"
#include "neuralnetworkoptions.h"

class NeuralNetwork;

class NeuralNetwork
{
private:
  class LayersAndNeuronsContainer
  {
  public:
    LayersAndNeuronsContainer(const LayersAndNeuronsContainer& src) noexcept :
      _offsets(src._offsets),
      _total_size(src._total_size),
      _data(src._data),
      _topology(src._topology)
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    }

    LayersAndNeuronsContainer(LayersAndNeuronsContainer&& src) noexcept :
      _offsets(std::move(src._offsets)),
      _total_size(src._total_size),
      _data(std::move(src._data)),
      _topology(std::move(src._topology))
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      src._total_size = 0;
    }

    LayersAndNeuronsContainer& operator=(const LayersAndNeuronsContainer& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      if(this != &src)
      {
        _offsets = src._offsets;
        _total_size = src._total_size;
        _data = src._data;
        _topology = src._topology;
      }
      return *this;
    }
 
    LayersAndNeuronsContainer& operator=(LayersAndNeuronsContainer&& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      if(this != &src)
      {
        _offsets = std::move(src._offsets);
        _total_size = src._total_size;
        _data = std::move(src._data);
        _topology = std::move(src._topology);
        src._total_size = 0;
      }
      return *this;
    }

    LayersAndNeuronsContainer(const std::vector<unsigned>& topology) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      const size_t topology_size = topology.size();
      _topology.reserve(topology_size);
      _offsets.reserve(topology_size);

      _topology.insert(_topology.begin(), topology.begin(), topology.end());
//      for (auto& topology : _topology)
//      {
//        ++topology;  //  bias
//      }

      // finaly populate the data
      const auto& size = _topology.size();
      _offsets.resize(size);
      _total_size = 0;
      for(size_t layer = 0; layer < size; ++layer)
      {
        _offsets[layer] = _total_size;
        _total_size+= _topology[layer];
      }
      _data.resize(_total_size);
    }

    LayersAndNeuronsContainer& operator=(const std::vector<std::vector<double>>& data)
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      assert(_topology.size() == data.size());
      for(size_t layer = 0; layer < data.size(); ++layer)
      {
        set(static_cast<unsigned>(layer), data[layer]);
      }
      return *this;
    }

    inline void zero()
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      std::fill(_data.begin(), _data.end(), 0.0);
    }

    inline void set( unsigned layer, unsigned neuron, double&& data)
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      ensure_size(layer, neuron);
      _data[_offsets[layer]+neuron] = std::move(data);
    }

    inline void set( unsigned layer, unsigned neuron, const double& data)
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      ensure_size(layer, neuron);
      _data[_offsets[layer]+neuron] = data;
    }
    
    void set(unsigned layer, const std::vector<double>& data)
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      assert(number_neurons(layer) == data.size());
      unsigned neuron = 0;
      const auto& layer_offset = _offsets[layer];
      for( const auto& d : data )
      {
        _data[layer_offset+neuron] = d;
        ++neuron;
      }
    }
    
    inline const double& get(unsigned layer, unsigned neuron) const noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      ensure_size(layer, neuron);
      return _data[_offsets[layer]+neuron];
    }

    std::vector<double> get_neurons(unsigned layer) const
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      std::vector<double> data;
      data.reserve(_topology[layer]);
      const auto layer_offset = _offsets[layer];
      for(size_t neuron = 0; neuron < _topology[layer]; ++neuron)
      {
        data.emplace_back(_data[layer_offset+neuron]);
      }
      return data;
    }

    size_t number_layers() const noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      return _topology.size();
    }

    size_t number_neurons(size_t layer) const noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      if(layer >= _topology.size())
      {
        return 0;
      }
      return _topology[layer];
    }
  private:
    #ifdef NDEBUG
    void ensure_size(size_t, size_t) const
    {
    }
    #else
    void ensure_size(size_t layer, size_t neuron) const
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      if (layer >= _topology.size() || neuron >= _topology[layer])
      {
        std::cerr << "The layer/neuron is out of bound!" << std::endl;
        throw std::invalid_argument("The layer/neuron is out of bound!");
      }
    }
    #endif

    std::vector<size_t> _offsets;
    size_t _total_size;
    alignas(32) std::vector<double> _data;
    std::vector<unsigned short> _topology;
  };

  class GradientsAndOutputs
  {
  public:
    GradientsAndOutputs() = delete;

    GradientsAndOutputs(const std::vector<unsigned>& topology) noexcept: 
      _outputs(topology),
      _gradients(topology)
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    }

    GradientsAndOutputs(const GradientsAndOutputs& src) noexcept:
      _outputs(src._outputs),
      _gradients(src._gradients)
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    }

    GradientsAndOutputs(GradientsAndOutputs&& src) noexcept: 
      _outputs(std::move(src._outputs)),
      _gradients(std::move(src._gradients))
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    }

    GradientsAndOutputs& operator=(const GradientsAndOutputs& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      if( &src != this)
      {
        _outputs = src._outputs;
        _gradients = src._gradients;
      }
      return *this;
    }
    GradientsAndOutputs& operator=(GradientsAndOutputs&& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      if( &src != this)
      {
        _outputs = std::move(src._outputs);
        _gradients = std::move(src._gradients);
      }
      return *this;
    }
    virtual ~GradientsAndOutputs() = default;

    void zero()
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      _outputs.zero();
      _gradients.zero();
    }

    [[nodiscard]] inline unsigned num_output_layers() const  noexcept
    { 
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      return static_cast<unsigned>(_outputs.number_layers());
    }

    [[nodiscard]] inline unsigned num_output_neurons(unsigned layer) const noexcept
    { 
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      return static_cast<unsigned>(_outputs.number_neurons(layer));
    }

    double get_gradient(unsigned layer, unsigned neuron) const
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      return _gradients.get(layer, neuron);
    }

    void set_gradient(unsigned layer, unsigned neuron, double gradient)
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      _gradients.set(layer, neuron, gradient);
    }

    std::vector<double> get_gradients(unsigned layer) const
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      return _gradients.get_neurons(layer);
    }

    void set_gradients(unsigned layer, const std::vector<double>& gradients)
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      _gradients.set(layer, gradients);
    }

    void set_gradients(const LayersAndNeuronsContainer& gradients)
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      _gradients = gradients;
    }

    void set_gradients(const std::vector<std::vector<double>>& gradients)
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      _gradients = gradients;
    }

    unsigned num_gradient_layers() const 
    { 
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      return static_cast<unsigned>(_gradients.number_layers());
    }

    unsigned num_gradient_neurons(unsigned layer) const 
    { 
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      return static_cast<unsigned>(_gradients.number_neurons(layer));
    }

    [[nodiscard]] inline std::vector<double> get_outputs(unsigned layer) const noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      return _outputs.get_neurons(layer);
    }

    [[nodiscard]] inline double get_output(unsigned layer, unsigned neuron) const noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
#ifdef _MSC_VER      
      if (_outputs.number_neurons(layer) == neuron)
#else
      if(__builtin_expect(_outputs.number_neurons(layer) == neuron, 0))
#endif
      {
        return 1.0; //  bias
      }      
      return _outputs.get(layer, neuron);
    }

    unsigned num_outputs(unsigned layer) const
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      return static_cast<unsigned>(_outputs.number_neurons(layer));
    }
    
    void set_outputs(unsigned layer, const std::vector<double>& outputs)
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      _outputs.set(layer, outputs);
    }
    
    [[nodiscard]] std::vector<double> output_back() const
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      const size_t size = _outputs.number_layers();
      if(size == 0)
      {
        std::cerr << "Trying to get the last output but none available!" << std::endl;
        throw std::invalid_argument("Trying to get the last output but none available!");
      }
      return _outputs.get_neurons(static_cast<unsigned>(size -1));
    }

  private:
    alignas(32) LayersAndNeuronsContainer _outputs;
    alignas(32) LayersAndNeuronsContainer _gradients;
  };

public:
  NeuralNetwork(const NeuralNetworkOptions& options);
  NeuralNetwork(const std::vector<unsigned>& topology, const activation::method& hidden_layer_activation, const activation::method& output_layer_activation);
  NeuralNetwork(const std::vector<Layer>& layers, const NeuralNetworkOptions& options, const std::map<NeuralNetworkOptions::ErrorCalculation, double>& errors);
  NeuralNetwork(const NeuralNetwork& src);
  NeuralNetwork& operator=(const NeuralNetwork&);

  virtual ~NeuralNetwork();

  void train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs);

  std::vector<std::vector<double>> think(const std::vector<std::vector<double>>& inputs) const;
  std::vector<double> think(const std::vector<double>& inputs) const;

  const std::vector<unsigned>& get_topology() const;
  const std::vector<Layer>& get_layers() const;
  const activation::method& get_output_activation_method() const;
  const activation::method& get_hidden_activation_method() const;

  NeuralNetworkHelper::NeuralNetworkHelperMetrics calculate_forecast_metric(NeuralNetworkOptions::ErrorCalculation error_type) const;
  std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> calculate_forecast_metrics(const std::vector<NeuralNetworkOptions::ErrorCalculation>& error_types) const;
  double get_learning_rate() const noexcept;

  bool has_training_data() const;

  inline NeuralNetworkOptions& options() noexcept { 
    MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    return _options;
  }
  inline const NeuralNetworkOptions& options() const noexcept { 
    MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    return _options;
  }

private:
  void calculate_back_propagation(
    std::vector<GradientsAndOutputs>& gradients,
    const std::vector<std::vector<double>>& outputs, 
    const Layers& layers) const;

  void calculate_back_propagation_input_layer(
    std::vector<GradientsAndOutputs>& gradients,
    const Layers& layers) const;

  void calculate_back_propagation_output_layer(
    std::vector<GradientsAndOutputs>& gradients,
    const std::vector<std::vector<double>>& outputs,
    const Layers& layers) const;

  void calculate_back_propagation_hidden_layers(
    std::vector<GradientsAndOutputs>& gradients,
    const Layers& layers) const;
  void calculate_forward_feed(
    std::vector<GradientsAndOutputs>& gradients_and_output,
    std::vector<HiddenStates>& hidden_states,
    const std::vector<std::vector<double>>& inputs, 
    const Layers& layers, 
    bool is_training) const;
  std::vector<GradientsAndOutputs> train_single_batch(
    const std::vector<std::vector<double>>::const_iterator inputs_begin, 
    const std::vector<std::vector<double>>::const_iterator outputs_begin,
    const size_t size
  ) const;

  void set_gradients_for_layer(std::vector<GradientsAndOutputs>& source, unsigned layer_number, const std::vector<std::vector<double>>& gradients) const;
  std::vector<std::vector<double>> get_outputs_for_layer(const std::vector<GradientsAndOutputs>& source, unsigned layer_number) const;
  std::vector<std::vector<double>> get_gradients_for_layer(const std::vector<GradientsAndOutputs>& source, unsigned layer_number) const;

  std::vector<double> calculate_weight_gradients(const unsigned batch_size, unsigned layer_number, unsigned neuron_number, const GradientsAndOutputs& source) const;
  std::vector<double> calculate_residual_projection_gradients(const unsigned batch_size, unsigned layer_number, unsigned neuron_number, const GradientsAndOutputs& source) const;

  void apply_weight_gradients(Layers& layers, const std::vector<std::vector<GradientsAndOutputs>>& batch_activation_gradients, double learning_rate, unsigned epoch) const;
  void apply_weight_gradients(Layers& layers, const std::vector<GradientsAndOutputs>& batch_activation_gradients, double learning_rate, unsigned epoch) const;

  Layer* get_residual_layer(Layers& layers, const GradientsAndOutputs& batch_activation_gradient, std::vector<double>& residual_output_values, const Layer& current_layer) const;

  std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> calculate_forecast_metrics(const std::vector<NeuralNetworkOptions::ErrorCalculation>& error_types, bool final_check) const;

  // Calculates the Mean Absolute Percentage Error (MAPE)
  double calculate_forecast_mape(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions, double epsilon = 1e-8) const;
  double calculate_forecast_smape(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions, double epsilon = 1e-8) const;

  // Calculates the Weighted Absolute Percentage Error(WAPE) for a batch of sequences.
  double calculate_forecast_wape(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions) const;

  // Error calculations
  // Todo this should be moved to a static class a passed as an object.
  // Todo: The user should be able to choose what error they want to use.
  // Todo: Should those be public so the called _could_ use them to compare a prediction?
  double calculate_error(NeuralNetworkOptions::ErrorCalculation error_type, const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;
  double calculate_huber_loss_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions, double delta = 1.0) const;
  double calculate_mae_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;
  double calculate_mse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;
  double calculate_rmse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;
  double calculate_nrmse_error(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions) const;
  double calculate_directional_accuracy(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions, 
    double neutral_tolerance = 0.001 // threshold below which movement is ignored
  ) const;
  double calculate_bce_loss(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions) const;

  void recreate_batch_from_indexes(NeuralNetworkHelper& neural_network_helper, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& shuffled_training_inputs, std::vector<std::vector<double>>& shuffled_training_outputs) const;
  void create_batch_from_indexes(const std::vector<size_t>& shuffled_indexes, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& shuffled_training_inputs, std::vector<std::vector<double>>& shuffled_training_outputs) const;
  void break_indexes(const std::vector<size_t>& indexes, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes) const;
  void create_shuffled_indexes(NeuralNetworkHelper& neural_network_helper, bool data_is_unique) const;
  void create_indexes(NeuralNetworkHelper& neural_network_helper, bool data_is_unique) const;

  double calculate_clipping_scale(const Layer& layer, unsigned int layer_number) const;

  double calculate_learning_rate(double learning_rate_base, double learning_rate_decay_rate, int epoch, int number_of_epoch, AdaptiveLearningRateScheduler& learning_rate_scheduler) const;
  double calculate_smooth_learning_rate_boost(int epoch, int total_epochs, double base_learning_rate) const;
  double calculate_learning_rate_warmup(int epoch, double completed_percent) const;

  bool CallCallback(const std::function<bool(NeuralNetworkHelper&)>& callback, SingleTaskQueue<bool>* callback_task) const;

  void log_training_info(
    const std::vector<std::vector<double>>& training_inputs,
    const std::vector<std::vector<double>>& training_outputs) const;

  std::vector<size_t> get_shuffled_indexes(size_t raw_size) const;

  mutable std::shared_mutex _mutex;

  double _learning_rate;
  Layers _layers;
  NeuralNetworkOptions _options;
  NeuralNetworkHelper* _neural_network_helper;
  std::map<NeuralNetworkOptions::ErrorCalculation, double> _saved_errors;

  Rng _rng;
};