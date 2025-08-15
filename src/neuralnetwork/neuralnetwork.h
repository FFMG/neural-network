#pragma once
#include <cassert>
#include <functional>
#include <shared_mutex>
#include <vector>

#include "./libraries/instrumentor.h"

#include "activation.h"
#include "layers.h"
#include "logger.h"
#include "neuron.h"
#include "optimiser.h"
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
      }
      return *this;
    }

    LayersAndNeuronsContainer(const std::vector<unsigned>& topology, bool shifted_by_one=false, bool add_bias=false) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      const size_t topology_size = topology.size();
      _topology.reserve(topology_size);
      _offsets.reserve(topology_size);

      if(shifted_by_one)
      {
        _topology.insert(_topology.begin(), topology.begin()+1, topology.end());
      }
      else
      {
        _topology.insert(_topology.begin(), topology.begin(), topology.end());
      }
      if(add_bias)
      {
        for (auto& t : _topology)
        {
          ++t;  //  bias
        }
      }

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
    inline void add(const LayersAndNeuronsContainer& container)
    {
      MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
      assert(_data.size() == container._data.size());
      for( size_t index = 0; index < _data.size(); ++index)
      {
        _data[index] += container._data[index];
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

    GradientsAndOutputs(const std::vector<unsigned>& topology, unsigned batch_size = 0) noexcept: 
      _outputs(topology, false, false),
      _gradients(topology, false, true),
      _batch_size(batch_size)
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    }

    GradientsAndOutputs(const GradientsAndOutputs& src) noexcept:
      _outputs(src._outputs),
      _gradients(src._gradients),
      _batch_size(src._batch_size)
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    }

    GradientsAndOutputs(GradientsAndOutputs&& src) noexcept: 
      _outputs(std::move(src._outputs)),
      _gradients(std::move(src._gradients)),
      _batch_size(src._batch_size)
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      src._batch_size = 0;
    }

    GradientsAndOutputs& operator=(const GradientsAndOutputs& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      if( &src != this)
      {
        _outputs = src._outputs;
        _gradients = src._gradients;
        _batch_size = src._batch_size;
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
        _batch_size = src._batch_size;
        src._batch_size = 0;
      }
      return *this;
    }
    virtual ~GradientsAndOutputs() = default;
    void zero()
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      _batch_size = 0;
      _outputs.zero();
      _gradients.zero();
    }
    void add(const GradientsAndOutputs& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      _outputs.add(src._outputs);
      _gradients.add(src._gradients);
    }
    
    [[nodiscard]] inline unsigned num_output_layers() const  noexcept
    { 
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      return static_cast<unsigned>(_outputs.number_layers());
    }

    [[nodiscard]] inline unsigned batch_size() const noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      return _batch_size;
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

    [[nodiscard]] inline double get_output(unsigned layer, unsigned neuron) const noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      if(_outputs.number_neurons(layer) == neuron)
      {
        return 1.0; //  bias
      }      
      return _outputs.get(layer, neuron);
    }

    unsigned num_outputs(unsigned layer) const
    {
      MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
      //  add the bias
      return static_cast<unsigned>(_outputs.number_neurons(layer) + 1);
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
    unsigned _batch_size;
  };

public:
  NeuralNetwork(const NeuralNetworkOptions& options);
  NeuralNetwork(const std::vector<unsigned>& topology, const activation::method& hidden_layer_activation, const activation::method& output_layer_activation, const Logger& logger);
  NeuralNetwork(const std::vector<Layer>& layers, const NeuralNetworkOptions& options);
  NeuralNetwork(const NeuralNetwork& src);
  NeuralNetwork& operator=(const NeuralNetwork&) = delete;

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
  double get_learning_rate() const;

  NeuralNetworkOptions& options() { 
    MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    return _options;
  }
  const NeuralNetworkOptions& options() const { 
    MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    return _options;
  }

private:
  void calculate_back_propagation(
    GradientsAndOutputs& gradients,
    const std::vector<double>& outputs, 
    const Layers& layers) const;
  void calculate_forward_feed(
    GradientsAndOutputs& gradients,
    const std::vector<double>& inputs, 
    const Layers& layers, 
    bool is_training) const;
  GradientsAndOutputs train_single_batch(
    const std::vector<std::vector<double>>::const_iterator inputs_begin, 
    const std::vector<std::vector<double>>::const_iterator outputs_begin,
    const size_t size
  ) const;

  std::vector<double> calculate_weight_gradients(unsigned layer_number, unsigned neuron_number, const GradientsAndOutputs& source) const;
  std::vector<double> calculate_residual_projection_gradients(unsigned layer_number, unsigned neuron_number, const GradientsAndOutputs& source) const;

  void apply_weight_gradients(Layers& layers, const std::vector<GradientsAndOutputs>& batch_activation_gradients, double learning_rate, unsigned epoch) const;
  void apply_weight_gradients(Layers& layers, const GradientsAndOutputs& batch_activation_gradient, double learning_rate, unsigned epoch) const;

  Layer* get_residual_layer(Layers& layers, const GradientsAndOutputs& batch_activation_gradient, std::vector<double>& residual_output_values, const Layer& current_layer) const;

  std::vector<double> calculate_output_gradients(const std::vector<double>& target_outputs, const std::vector<double>& given_outputs, const Layer& output_layer) const;

  std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> calculate_forecast_metrics(const std::vector<NeuralNetworkOptions::ErrorCalculation>& error_types, bool final_check) const;

  // Calculates the Mean Absolute Percentage Error (MAPE)
  double calculate_forecast_mape(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions, double epsilon = 1e-8) const;
  double calculate_forecast_smape(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions, double epsilon = 1e-8) const;

  // Error calculations
  // Todo this should be moved to a static class a passed as an object.
  // Todo: The user should be able to choose what error they want to use.
  // Todo: Should those be public so the called _could_ use them to compare a prediction?
  double calculate_error(NeuralNetworkOptions::ErrorCalculation error_type, const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;
  double calculate_huber_loss_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions, double delta = 1.0) const;
  double calculate_mae_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;
  double calculate_mse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;
  double calculate_rmse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions ) const;

  void recreate_batch_from_indexes(NeuralNetworkHelper& neural_network_helper, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& shuffled_training_inputs, std::vector<std::vector<double>>& shuffled_training_outputs) const;
  void create_batch_from_indexes(const std::vector<size_t>& shuffled_indexes, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& shuffled_training_inputs, std::vector<std::vector<double>>& shuffled_training_outputs) const;
  void break_shuffled_indexes(const std::vector<size_t>& shuffled_indexes, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes) const;
  void create_shuffled_indexes(NeuralNetworkHelper& neural_network_helper, bool data_is_unique) const;

  double calculate_clipping_scale(const Layer& layer, unsigned int layer_number) const;

  double calculate_learning_rate(double learning_rate_base, double learning_rate_decay_rate, int epoch, int number_of_epoch, AdaptiveLearningRateScheduler& learning_rate_scheduler) const;
  double calculate_smooth_learning_rate_boost(int epoch, int total_epochs, double base_learning_rate) const;
  double calculate_learning_rate_warmup(int epoch, double completed_percent) const;

  bool CallCallback(const std::function<bool(NeuralNetworkHelper&)>& callback, SingleTaskQueue<bool>* callback_task) const;

  const Logger& logger() const;

  void log_training_info(
    const std::vector<std::vector<double>>& training_inputs,
    const std::vector<std::vector<double>>& training_outputs) const;

  void dump_layer_info() const;

  std::vector<size_t> get_shuffled_indexes(size_t raw_size) const;

  double _learning_rate;
  Layers _layers;
  mutable std::shared_mutex _mutex;

  NeuralNetworkOptions _options;
  NeuralNetworkHelper* _neural_network_helper;
};