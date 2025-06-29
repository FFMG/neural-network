#pragma once
#include <cassert>
#include <functional>
#include <vector>

#include "./libraries/instrumentor.h"

#include "activation.h"
#include "layer.h"
#include "logger.h"
#include "neuron.h"
#include "taskqueue.h"

class NeuralNetwork
{
private:
  template <typename T>
  class LayersAndNeurons
  {
  public:
    LayersAndNeurons(const LayersAndNeurons& src) noexcept :
      _offsets(src._offsets),
      _total_size(src._total_size),
      _data(src._data),
      _topology(src._topology)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    }

    LayersAndNeurons(LayersAndNeurons&& src) noexcept :
      _offsets(std::move(src._offsets)),
      _total_size(src._total_size),
      _data(std::move(src._data)),
      _topology(std::move(src._topology))
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    }

    LayersAndNeurons& operator=(const LayersAndNeurons& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      if(this != &src)
      {
        _offsets = src._offsets;
        _total_size = src._total_size;
        _data = src._data;
        _topology = src._topology;
      }
      return *this;
    }

    LayersAndNeurons& operator=(LayersAndNeurons&& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      if(this != &src)
      {
        _offsets = std::move(src._offsets);
        _total_size = src._total_size;
        _data = std::move(src._data);
        _topology = std::move(src._topology);
      }
      return *this;
    }

    LayersAndNeurons(const std::vector<unsigned>& topology, bool shifted_by_one=false, bool add_bias=false) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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

    LayersAndNeurons& operator=(const std::vector<std::vector<T>>& data)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      assert(_topology.size() == data.size());
      for(size_t layer = 0; layer < data.size(); ++layer)
      {
        set(static_cast<unsigned>(layer), data[layer]);
      }
      return *this;
    }

    inline void set( unsigned layer, unsigned neuron, T&& data)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      ensure_size(layer, neuron);
      _data[_offsets[layer]+neuron] = std::move(data);
    }

    inline void set( unsigned layer, unsigned neuron, const T& data)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      ensure_size(layer, neuron);
      _data[_offsets[layer]+neuron] = data;
    }

    inline void set(unsigned layer, const std::vector<T>& data)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      assert(number_neurons(layer) == data.size());
      unsigned neuron = 0;
      const auto& layer_offset = _offsets[layer];
      for( const auto& d : data )
      {
        _data[layer_offset+neuron] = d;
        ++neuron;
      }
    }
    
    inline const T& get(unsigned layer, unsigned neuron) const
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      ensure_size(layer, neuron);
      return _data[_offsets[layer]+neuron];
    }

    std::vector<T> get_neurons(unsigned layer) const
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      std::vector<T> data;
      data.reserve(_topology[layer]);
      const auto layer_offset = _offsets[layer];
      for(size_t neuron = 0; neuron < _topology[layer]; ++neuron)
      {
        data.emplace_back(_data[layer_offset+neuron]);
      }
      return data;
    }

    size_t number_layers() const
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      return _topology.size();
    }

    size_t number_neurons(size_t layer) const
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      if(layer >= _topology.size())
      {
        return 0;
      }
      return _topology[layer];
    }
  private:
    #ifdef NDEBUG
    void ensure_size(size_t layer, size_t neuron) const
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      if(layer >= _topology.size() || neuron >= _topology[layer])
      {
        std::cerr << "The layer/neuron is out of bound!" << std::endl;
        throw new std::invalid_argument("The layer/neuron is out of bound!");
      }
    }
    #else
    void ensure_size(size_t, size_t) const
    {
    }
    #endif

    std::vector<size_t> _offsets;
    size_t _total_size;
    std::vector<T> _data;
    std::vector<unsigned short> _topology;
  };

  class GradientsAndOutputs
  {
  public:
    GradientsAndOutputs() = delete;

    GradientsAndOutputs(const std::vector<unsigned>& topology) noexcept: 
      _batch_size(0),
      _outputs(topology, false, false),
      _gradients(topology, false, true),
      _gradients_and_outputs(topology, true, true)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    }

    GradientsAndOutputs(const GradientsAndOutputs& src) noexcept:
      _batch_size(src._batch_size),
      _outputs(src._outputs),
      _gradients(src._gradients),
      _gradients_and_outputs(src._gradients_and_outputs)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    }

    GradientsAndOutputs(GradientsAndOutputs&& src) noexcept: 
      _batch_size(src._batch_size),
      _outputs(std::move(src._outputs)),
      _gradients(std::move(src._gradients)),
      _gradients_and_outputs(std::move(src._gradients_and_outputs))
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      src._batch_size = 0;
    }

    GradientsAndOutputs& operator=(const GradientsAndOutputs& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      if( &src != this)
      {
        _batch_size = src._batch_size;
        _outputs = src._outputs;
        _gradients = src._gradients;
        _gradients_and_outputs = src._gradients_and_outputs;
      }
      return *this;
    }
    GradientsAndOutputs& operator=(GradientsAndOutputs&& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      if( &src != this)
      {
        _batch_size = src._batch_size;
        _outputs = std::move(src._outputs);
        _gradients = std::move(src._gradients);
        _gradients_and_outputs = std::move(src._gradients_and_outputs);
        src._batch_size = 0;
      }
      return *this;
    }
    virtual ~GradientsAndOutputs() = default;

    unsigned num_output_layers() const 
    { 
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      return static_cast<unsigned>(_outputs.number_layers());
    }

    unsigned num_output_neurons(unsigned layer) const 
    { 
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      return static_cast<unsigned>(_outputs.number_neurons(layer));
    }

    unsigned num_gradients_and_outputs_layers() const 
    { 
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      return static_cast<unsigned>(_gradients_and_outputs.number_layers());
    }

    unsigned num_gradients_and_outputs_neurons(unsigned layer) const 
    { 
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      return static_cast<unsigned>(_gradients_and_outputs.number_neurons(layer));
    }

    inline const LayersAndNeurons<std::vector<double>>& get_gradients_and_outputs() const
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      return _gradients_and_outputs;
    } 

    inline void set_gradients_and_outputs(unsigned layer, unsigned neuron, const std::vector<double>& data)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      return _gradients_and_outputs.set(layer, neuron, data);
    } 

    double get_gradient(unsigned layer, unsigned neuron) const
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      if(_batch_size == 0 )
      {
        return 0.0;
      }
      return _gradients.get(layer, neuron)/ static_cast<double>(_batch_size);
    }

    void set_gradient(unsigned layer, unsigned neuron, double gradient)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      // if we are calling 'set' then it is for a single batch
      // otherwise call the add_outputs_gradients(...)
      assert(_batch_size == 0 || _batch_size == 1);
      _gradients.set(layer, neuron, gradient);
      _batch_size = 1;
    }

    void set_gradients(unsigned layer, const std::vector<double>& gradients)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      _gradients.set(layer, gradients);
      _batch_size = 1;
    }

    void set_gradients(const LayersAndNeurons<double>& gradients)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      _gradients = gradients;
      _batch_size = 1;
    }

    void set_gradients(const std::vector<std::vector<double>>& gradients)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      // if we are calling 'set' then it is for a single batch
      // otherwise call the add_outputs_gradients(...)
      assert(_batch_size == 0 || _batch_size == 1);
      _gradients = gradients;
      _batch_size = 1;
    }

    unsigned num_gradient_layers() const 
    { 
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      return static_cast<unsigned>(_gradients.number_layers());
    }

    unsigned num_gradient_neurons(unsigned layer) const 
    { 
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      return static_cast<unsigned>(_gradients.number_neurons(layer));
    }

    double get_output(unsigned layer, unsigned neuron) const
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      if(_outputs.number_neurons(layer) == neuron)
      {
        return 1.0; //  bias
      }      
      return _outputs.get(layer, neuron);
    }

    unsigned num_outputs(unsigned layer) const
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      //  add the bias
      return static_cast<unsigned>(_outputs.number_neurons(layer) + 1);
    }
    
    void set_outputs(unsigned layer, const std::vector<double>& outputs)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      _outputs.set(layer, outputs);
    }

    std::vector<double> output_back() const
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      const size_t size = _outputs.number_layers();
      if(size == 0)
      {
        std::cerr << "Trying to get the last output but none available!" << std::endl;
        throw new std::invalid_argument("Trying to get the last output but none available!");
      }
      return _outputs.get_neurons(static_cast<unsigned>(size -1));
    }

    void zero()
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      for( size_t layer = 0; layer < num_gradient_layers(); ++layer)
      {
        for( size_t neuron = 0; neuron < num_gradient_neurons(static_cast<unsigned>(layer)); ++layer)
        {
          _gradients.set(static_cast<unsigned>(layer), static_cast<unsigned>(neuron), 0);
        }
      }
      for( size_t layer = 0; layer < num_output_layers(); ++layer)
      {
        for( size_t neuron = 0; neuron < num_output_neurons(static_cast<unsigned>(layer)); ++layer)
        {
          _outputs.set(static_cast<unsigned>(layer), static_cast<unsigned>(neuron), 0);
        }
      }
      for( size_t layer = 0; layer < num_gradients_and_outputs_layers(); ++layer)
      {
        for( size_t neuron = 0; neuron < num_gradients_and_outputs_neurons(static_cast<unsigned>(layer)); ++layer)
        {
          _gradients_and_outputs.set(static_cast<unsigned>(layer), static_cast<unsigned>(neuron), {});
        }
      }
      _batch_size = 0;
    }

  private:
    int _batch_size;
    LayersAndNeurons<double> _outputs;
    LayersAndNeurons<double> _gradients;
    LayersAndNeurons<std::vector<double>> _gradients_and_outputs;
  };

public:
  NeuralNetwork(const std::vector<unsigned>& topology, const activation::method& hidden_layer_activation, const activation::method& output_layer_activation, const Logger& logger);
  NeuralNetwork(const std::vector<Layer>& layers, const activation::method& hidden_layer_activation, const activation::method& output_layer_activation, const Logger& logger, long double error, long double mean_absolute_percentage_error);
  NeuralNetwork(const NeuralNetwork& src);
  NeuralNetwork& operator=(const NeuralNetwork&) = delete;

  virtual ~NeuralNetwork() = default;

  void train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, double learning_rate, int number_of_epoch, int batch_size = 1, bool data_is_unique = true, const std::function<bool(int, int, NeuralNetwork&)>& progress_callback = nullptr);

  std::vector<std::vector<double>> think(
    const std::vector<std::vector<double>>& inputs
  ) const;
  std::vector<double> think(
    const std::vector<double>& inputs
  ) const;

  const std::vector<unsigned>& get_topology() const;
  const std::vector<Layer>& get_layers() const;
  const activation::method& get_output_activation_method() const;
  const activation::method& get_hidden_activation_method() const;
  long double get_error() const;
  long double get_mean_absolute_percentage_error() const;

private:
  GradientsAndOutputs calculate_forward_feed(const std::vector<double>& inputs, const std::vector<Layer>& layers) const;
  std::vector<GradientsAndOutputs> train_single_batch(
    const std::vector<std::vector<double>>::const_iterator inputs_begin, 
    const std::vector<std::vector<double>>::const_iterator outputs_begin,
    const size_t size
  ) const;
  std::vector<GradientsAndOutputs> calculate_forward_feed(
    const std::vector<std::vector<double>>::const_iterator inputs_begin, 
    const std::vector<std::vector<double>>::const_iterator inputs_end, 
    const std::vector<Layer>& layers) const;

  void calculate_batch_back_propagation(
    const std::vector<std::vector<double>>::const_iterator outputs_begin, 
    const size_t outputs_size, 
    std::vector<GradientsAndOutputs>& batch_given_outputs, 
    const std::vector<Layer>& layers) const;
  static void calculate_batch_back_propagation_gradients(
    const std::vector<std::vector<double>>::const_iterator outputs_begin, 
    const size_t outputs_size, 
    std::vector<GradientsAndOutputs>& layers_given_outputs, 
    const std::vector<Layer>& layers);

  void update_layers_with_gradients(const LayersAndNeurons<std::vector<double>>& activation_gradients, std::vector<Layer>& layers, double learning_rate) const;
  void update_layers_with_gradients(const std::vector<std::vector<GradientsAndOutputs>>& batch_activation_gradients, std::vector<Layer>& layers, double learning_rate) const;

  void average_batch_gradients_with_averages(const GradientsAndOutputs& activation_gradients, const LayersAndNeurons<double>& averages, LayersAndNeurons<std::vector<double>>& gradients_and_outputs) const;
  void recalculate_gradient_avergages(const std::vector<std::vector<GradientsAndOutputs>>& epoch_gradients_outputs, LayersAndNeurons<double>& averages) const;
  
  static void calculate_back_propagation_gradients(const std::vector<double>& target_outputs, GradientsAndOutputs& layers_given_outputs, const std::vector<Layer>& layers);
  static void calculate_batch_back_propagation_gradients(const std::vector<std::vector<double>>& target_outputs, std::vector<GradientsAndOutputs>& layers_given_outputs, const std::vector<Layer>& layers);

  static std::vector<double> caclulate_output_gradients(const std::vector<double>& target_outputs, const std::vector<double>& given_outputs, const Layer& output_layer);

  // Calculates the Mean Absolute Percentage Error (MAPE)
  double calculate_mape(const std::vector<double>& ground_truth, const std::vector<double>& predictions) const;
  double calculate_mape(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;

  double calculate_smape(const std::vector<double>& ground_truth, const std::vector<double>& predictions) const;
  double calculate_smape(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;

  void update_error_and_percentage_error(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, int batch_size, std::vector<Layer>& layers, TaskQueuePool<std::vector<std::vector<double>>>* errorPool);

  // Error calculations
  // Todo this should be moved to a static class a passed as an object.
  // Todo: The user should be able to choose what error they want to use.
  // Todo: Should those be public so the called _could_ use them to compare a prediction?
  double calculate_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;
  double calculate_huber_loss(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions, double delta = 1.0) const;
  double calculate_mae_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;
  double calculate_mse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;
  double calculate_rmse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions ) const;

  void recreate_batch_from_indexes(const std::vector<size_t>& shuffled_indexes, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& shuffled_training_inputs, std::vector<std::vector<double>>& shuffled_training_outputs) const;
  void create_batch_from_indexes(const std::vector<size_t>& shuffled_indexes, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& shuffled_training_inputs, std::vector<std::vector<double>>& shuffled_training_outputs) const;
  void break_shuffled_indexes(const std::vector<size_t>& shuffled_indexes, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes) const;
  void create_shuffled_indexes(size_t raw_size, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes) const;

  void log_training_info(
    double learning_rate,
    const std::vector<std::vector<double>>& training_inputs,
    const std::vector<std::vector<double>>& training_outputs,
    const std::vector<size_t>& training_indexes, const std::vector<size_t>& checking_indexes, const std::vector<size_t>& final_check_indexes) const;

  std::vector<size_t> get_shuffled_indexes(size_t raw_size) const;

  long double _error;
  long double _mean_absolute_percentage_error;
  std::vector<unsigned> _topology;
  std::vector<Layer> _layers;

  const activation::method _hidden_activation_method;
  const activation::method _output_activation_method;
  
  Logger _logger;
};