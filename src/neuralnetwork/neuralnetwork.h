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

class NeuralNetwork;
class NeuralNetworkOptions
{
private:
  NeuralNetworkOptions(const std::vector<unsigned>& topology) : 
    _topology(topology),
    _hidden_activation(activation::method::sigmoid),
    _output_activation(activation::method::sigmoid),
    _learning_rate(0.15),
    _number_of_epoch(1000),
    _batch_size(1),
    _data_is_unique(true),
    _progress_callback(nullptr),
    _logger(Logger::LogLevel::None),
    _number_of_threads(0),
    _learning_rate_decay_rate(0.0),
    _error_calculation(ErrorCalculation::none),
    _forecast_accuracy(ForecastAccuracy::none),
    _adaptive_learning_rate(false)
  {
  }

public:
  enum class ForecastAccuracy
  {
    none,
    mape,
    smape,
  };

  enum class ErrorCalculation
  {
    none,
    huber_loss,
    mae,
    mse,
    rmse
  };

  NeuralNetworkOptions(const NeuralNetworkOptions& nno) noexcept
  {
    *this = nno;
  }

  NeuralNetworkOptions(NeuralNetworkOptions&& nno) noexcept
  {
    *this = std::move(nno);
  }
  
  NeuralNetworkOptions& operator=(const NeuralNetworkOptions& nno) noexcept
  {
    if( this != &nno)
    {
      _topology = nno._topology;
      _hidden_activation = nno._hidden_activation;
      _output_activation = nno._output_activation;
      _learning_rate = nno._learning_rate;
      _number_of_epoch = nno._number_of_epoch;
      _batch_size = nno._batch_size;
      _data_is_unique = nno._data_is_unique;
      _progress_callback = nno._progress_callback;
      _logger = nno._logger;
      _number_of_threads = nno._number_of_threads;
      _learning_rate_decay_rate = nno._learning_rate_decay_rate;
      _error_calculation = nno._error_calculation;
      _forecast_accuracy = nno._forecast_accuracy;
      _adaptive_learning_rate = nno._adaptive_learning_rate;
    }
    return *this;
  }
  NeuralNetworkOptions& operator=(NeuralNetworkOptions&& nno) noexcept
  {
    if( this != &nno)
    {
      _topology = std::move(nno._topology);
      _hidden_activation = nno._hidden_activation;
      _output_activation = nno._output_activation;
      _learning_rate = nno._learning_rate;
      _number_of_epoch = nno._number_of_epoch;
      _batch_size = nno._batch_size;
      _data_is_unique = nno._data_is_unique;
      _progress_callback = nno._progress_callback;
      _logger = nno._logger;
      _number_of_threads = nno._number_of_threads;
      _learning_rate_decay_rate = nno._learning_rate_decay_rate;
      _error_calculation = nno._error_calculation;
      _forecast_accuracy = nno._forecast_accuracy;
      _adaptive_learning_rate = nno._adaptive_learning_rate;

      nno._number_of_epoch = 0;
      nno._batch_size = 0;
      nno._learning_rate = 0.00;
      nno._data_is_unique = false;
      nno._error_calculation = ErrorCalculation::none;
      nno._forecast_accuracy = ForecastAccuracy::none;
    }
    return *this;
  }

  NeuralNetworkOptions& with_learning_rate(double learning_rate)
  {
    _learning_rate = learning_rate;
    return *this;
  }
  NeuralNetworkOptions& with_hidden_activation_method(const activation::method& activation)
  {
    _hidden_activation = activation;
    return *this;
  }
  NeuralNetworkOptions& with_output_activation_method(const activation::method& activation)
  {
    _output_activation = activation;
    return *this;
  }
  NeuralNetworkOptions& with_number_of_epoch(int number_of_epoch)
  {
    _number_of_epoch = number_of_epoch;
    return *this;
  }
  NeuralNetworkOptions& with_batch_size(int batch_size)
  {
    _batch_size = batch_size;
    return *this;
  }
  NeuralNetworkOptions& with_data_is_unique(bool data_is_unique)
  {
    // unique training data means that we cannot have
    // data split for epoch error checking and final error checking.
    _data_is_unique = data_is_unique;
    return *this;
  }
  NeuralNetworkOptions& with_progress_callback(const std::function<bool(int, int, NeuralNetwork&)>& progress_callback)
  {
    _progress_callback = progress_callback;
    return *this;
  }
  NeuralNetworkOptions& with_logger(const Logger& logger)
  {
    _logger = logger;
    return *this;
  }
  NeuralNetworkOptions& with_number_of_threads(int number_of_threads)
  {
    _number_of_threads = number_of_threads <= 0 ? 0 : number_of_threads;
    return *this;
  }
  NeuralNetworkOptions& with_learning_rate_decay_rate(double learning_rate_decay_rate)
  {
    _learning_rate_decay_rate = learning_rate_decay_rate;
    return *this;
  }
  NeuralNetworkOptions& with_no_error_calculation()
  {
    return with_error_calculation(ErrorCalculation::none);
  }
  NeuralNetworkOptions& with_huber_loss_error_calculation()
  {
    return with_error_calculation(ErrorCalculation::huber_loss);
  }
  NeuralNetworkOptions& with_mae_error_calculation()
  {
    return with_error_calculation(ErrorCalculation::mae);
  }
  NeuralNetworkOptions& with_mse_error_calculation()
  {
    return with_error_calculation(ErrorCalculation::mse);
  }
  NeuralNetworkOptions& with_rmse_error_calculation()
  {
    return with_error_calculation(ErrorCalculation::rmse);
  }
  NeuralNetworkOptions& with_error_calculation(const ErrorCalculation& error_calculation)
  {
    _error_calculation = error_calculation;
    return *this;
  }
  NeuralNetworkOptions& with_no_forecast_accuracy()
  {
    return with_forecast_accuracy(ForecastAccuracy::none);
  }
  NeuralNetworkOptions& with_mape_forecast_accuracy()
  {
    return with_forecast_accuracy(ForecastAccuracy::mape);
  }
  NeuralNetworkOptions& with_smape_forecast_accuracy()
  {
    return with_forecast_accuracy(ForecastAccuracy::smape);
  }
  NeuralNetworkOptions& with_forecast_accuracy(const ForecastAccuracy& forecast_accuracy)
  {
    _forecast_accuracy = forecast_accuracy;
    return *this;
  }
  NeuralNetworkOptions& with_adaptive_learning_rates(bool adaptive_learning_rate)
  {
    _adaptive_learning_rate = adaptive_learning_rate;
    return *this;
  }

  NeuralNetworkOptions& build()
  {
    if(topology().size() < 2)
    {
      logger().log_error("The topology is not value, you must have at least 2 layers.");
      throw std::invalid_argument("The topology is not value, you must have at least 2 layers.");
    }
    if(number_of_threads() > 0 && batch_size() <= 1 )
    {
      logger().log_warning("Because the batch size is 1, the number of threads is ignored." );
    }
    if (learning_rate_decay_rate() < 0)
    {
      logger().log_error("The learning rate decay rate cannot be negative!");
      throw std::invalid_argument("The learning rate decay rate cannot be negative!");
    }
    return *this;
  }
  
  static NeuralNetworkOptions create(const std::vector<Layer>& layers)
  {
    auto topology = std::vector<unsigned>();
    topology.reserve(layers.size());
    for (const auto& layer : layers)
    {
      // remove the bias Neuron.
      topology.emplace_back(layer.size() -1);
    }
    return create(topology);
  }

  static NeuralNetworkOptions create(const std::vector<unsigned>& topology)
  {
    return NeuralNetworkOptions(topology)
      .with_learning_rate(0.1)
      .with_hidden_activation_method(activation::method::sigmoid)
      .with_output_activation_method(activation::method::sigmoid)
      .with_number_of_epoch(1000)
      .with_batch_size(1)
      .with_data_is_unique(true)
      .with_progress_callback(nullptr)
      .with_learning_rate_decay_rate(0.0)
      .with_rmse_error_calculation()
      .with_mape_forecast_accuracy()
      .with_adaptive_learning_rates(false);
  }

  inline const std::vector<unsigned>& topology() const { return _topology;}
  inline const activation::method& hidden_activation_method() const{ return _hidden_activation;}
  inline const activation::method& output_activation_method() const{ return _output_activation;}
  inline double learning_rate() const{ return _learning_rate;}
  inline int number_of_epoch() const { return _number_of_epoch;}
  inline int batch_size() const { return _batch_size; }
  inline bool data_is_unique() const { return _data_is_unique; }
  inline const std::function<bool(int, int, NeuralNetwork&)>& progress_callback() const{ return _progress_callback; }
  inline const Logger& logger() const { return _logger; }
  inline int number_of_threads() const {return _number_of_threads; }
  inline double learning_rate_decay_rate() const { return _learning_rate_decay_rate; }
  inline const ErrorCalculation& error_calculation() const { return _error_calculation; }
  inline const ForecastAccuracy& forecast_accuracy() const { return _forecast_accuracy; }
  inline bool adaptive_learning_rate() const { return _adaptive_learning_rate; }

private:
  std::vector<unsigned> _topology;
  activation::method _hidden_activation;
  activation::method _output_activation;
  double _learning_rate;
  int _number_of_epoch;
  int _batch_size;
  bool _data_is_unique;
  std::function<bool(int, int, NeuralNetwork&)> _progress_callback;
  Logger _logger;
  int _number_of_threads;
  double _learning_rate_decay_rate;
  ErrorCalculation _error_calculation;
  ForecastAccuracy _forecast_accuracy;
  bool _adaptive_learning_rate;
};

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

    void set(unsigned layer, const std::vector<T>& data)
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
    void ensure_size(size_t, size_t) const
    {
    }
    #else
    void ensure_size(size_t layer, size_t neuron) const
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      if (layer >= _topology.size() || neuron >= _topology[layer])
      {
        std::cerr << "The layer/neuron is out of bound!" << std::endl;
        throw new std::invalid_argument("The layer/neuron is out of bound!");
      }
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
      _gradients(topology, false, true)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    }

    GradientsAndOutputs(const GradientsAndOutputs& src) noexcept:
      _batch_size(src._batch_size),
      _outputs(src._outputs),
      _gradients(src._gradients)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    }

    GradientsAndOutputs(GradientsAndOutputs&& src) noexcept: 
      _batch_size(src._batch_size),
      _outputs(std::move(src._outputs)),
      _gradients(std::move(src._gradients))
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

  private:
    int _batch_size;
    LayersAndNeurons<double> _outputs;
    LayersAndNeurons<double> _gradients;
  };

public:
  NeuralNetwork(const NeuralNetworkOptions options);
  NeuralNetwork(const std::vector<unsigned>& topology, const activation::method& hidden_layer_activation, const activation::method& output_layer_activation, const Logger& logger);
  NeuralNetwork(const std::vector<Layer>& layers, const activation::method& hidden_layer_activation, const activation::method& output_layer_activation, const Logger& logger, long double error, long double mean_absolute_percentage_error);
  NeuralNetwork(const NeuralNetwork& src);
  NeuralNetwork& operator=(const NeuralNetwork&) = delete;

  virtual ~NeuralNetwork() = default;

  void train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs);

  std::vector<std::vector<double>> think(const std::vector<std::vector<double>>& inputs) const;
  std::vector<double> think(const std::vector<double>& inputs) const;

  const std::vector<unsigned>& get_topology() const;
  const std::vector<Layer>& get_layers() const;
  const activation::method& get_output_activation_method() const;
  const activation::method& get_hidden_activation_method() const;
  long double get_error() const;
  long double get_mean_absolute_percentage_error() const;

  NeuralNetworkOptions& options() { return _options;}
  const NeuralNetworkOptions& options() const { return _options;}

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

  double calculate_forecast_accuracy(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;

  // Calculates the Mean Absolute Percentage Error (MAPE)
  double calculate_forecast_accuracy_mape(const std::vector<double>& ground_truth, const std::vector<double>& predictions) const;
  double calculate_forecast_accuracy_mape(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;

  double calculate_forecast_accuracy_smape(const std::vector<double>& ground_truth, const std::vector<double>& predictions) const;
  double calculate_forecast_accuracy_smape(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;

  void update_error_and_percentage_error(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, int batch_size, std::vector<Layer>& layers, TaskQueuePool<std::vector<std::vector<double>>>* errorPool);

  // Error calculations
  // Todo this should be moved to a static class a passed as an object.
  // Todo: The user should be able to choose what error they want to use.
  // Todo: Should those be public so the called _could_ use them to compare a prediction?
  double calculate_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;
  double calculate_huber_loss_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions, double delta = 1.0) const;
  double calculate_mae_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;
  double calculate_mse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;
  double calculate_rmse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions ) const;

  void recreate_batch_from_indexes(const std::vector<size_t>& shuffled_indexes, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& shuffled_training_inputs, std::vector<std::vector<double>>& shuffled_training_outputs) const;
  void create_batch_from_indexes(const std::vector<size_t>& shuffled_indexes, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& shuffled_training_inputs, std::vector<std::vector<double>>& shuffled_training_outputs) const;
  void break_shuffled_indexes(const std::vector<size_t>& shuffled_indexes, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes) const;
  void create_shuffled_indexes(size_t raw_size, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes) const;

  void log_training_info(
    const std::vector<std::vector<double>>& training_inputs,
    const std::vector<std::vector<double>>& training_outputs,
    const std::vector<size_t>& training_indexes, const std::vector<size_t>& checking_indexes, const std::vector<size_t>& final_check_indexes) const;

  std::vector<size_t> get_shuffled_indexes(size_t raw_size) const;

  long double _error;
  long double _mean_absolute_percentage_error;
  std::vector<Layer> _layers;

  NeuralNetworkOptions _options;
};