#pragma once
#include <cassert>
#include <functional>
#include <shared_mutex>
#include <vector>

#include "./libraries/instrumentor.h"

#include "activation.h"
#include "layer.h"
#include "logger.h"
#include "neuron.h"
#include "optimiser.h"
#include "taskqueue.h"

class NeuralNetwork;
class NeuralNetworkHelper;

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
    _adaptive_learning_rate(false),
    _optimiser_type(OptimiserType::SGD),
    _learning_rate_restart_rate(1),
    _learning_rate_restart_boost(1)
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
    if (this != &nno)
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
      _adaptive_learning_rate = nno._adaptive_learning_rate;
      _optimiser_type = nno._optimiser_type;
      _learning_rate_restart_rate = nno._learning_rate_restart_rate;
      _learning_rate_restart_boost = nno._learning_rate_restart_boost;
    }
    return *this;
  }

  NeuralNetworkOptions& operator=(NeuralNetworkOptions&& nno) noexcept
  {
    if (this != &nno)
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
      _adaptive_learning_rate = nno._adaptive_learning_rate;
      _optimiser_type = nno._optimiser_type;
      _learning_rate_restart_rate = nno._learning_rate_restart_rate;
      _learning_rate_restart_boost = nno._learning_rate_restart_boost;

      nno._number_of_epoch = 0;
      nno._batch_size = 0;
      nno._learning_rate = 0.00;
      nno._data_is_unique = false;
      nno._optimiser_type = OptimiserType::None;
    }
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

  NeuralNetworkOptions& with_progress_callback(const std::function<bool(NeuralNetworkHelper&)>& progress_callback)
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
  NeuralNetworkOptions& with_learning_rate(double learning_rate)
  {
    _learning_rate = learning_rate;
    return *this;
  }
  NeuralNetworkOptions& with_learning_rate_decay_rate(double learning_rate_decay_rate)
  {
    _learning_rate_decay_rate = learning_rate_decay_rate;
    return *this;
  }
  NeuralNetworkOptions& with_learning_rate_boost_rate(double every_percent, double restart_boost)
  {
    _learning_rate_restart_rate = every_percent;
    _learning_rate_restart_boost = restart_boost;
    return *this;
  }
  NeuralNetworkOptions& with_adaptive_learning_rates(bool adaptive_learning_rate)
  {
    _adaptive_learning_rate = adaptive_learning_rate;
    return *this;
  }
  NeuralNetworkOptions& with_optimiser_type(OptimiserType optimiser_type)
  {
    _optimiser_type = optimiser_type;
    return *this;
  }

  NeuralNetworkOptions& build()
  {
    if (topology().size() < 2)
    {
      logger().log_error("The topology is not value, you must have at least 2 layers.");
      throw std::invalid_argument("The topology is not value, you must have at least 2 layers.");
    }
    if (number_of_threads() > 0 && batch_size() <= 1)
    {
      logger().log_warning("Because the batch size is 1, the number of threads is ignored.");
    }
    if (learning_rate_decay_rate() < 0)
    {
      logger().log_error("The learning rate decay rate cannot be negative!");
      throw std::invalid_argument("The learning rate decay rate cannot be negative!");
    }
    if (learning_rate_decay_rate() >= 1.0)
    {
      logger().log_error("The learning rate decay rate cannot be more than 1!");
      throw std::invalid_argument("The learning rate decay rate cannot be more than 1!");
    }
    if (learning_rate_restart_rate() <= 0.0 || learning_rate_restart_rate() > 100)
    {
      logger().log_error("The learning rate has to be between 0% and 100%!");
      throw std::invalid_argument("The learning rate has to be between 0% and 100%!");
    }
    if (learning_rate_restart_boost() < 1.0)
    {
      logger().log_error("The learning rate restart boost cannot be less than 1!");
      throw std::invalid_argument("The learning rate restart boost cannot be less than 1!");
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
      topology.emplace_back(layer.size() - 1);
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
      .with_adaptive_learning_rates(false)
      .with_optimiser_type(OptimiserType::SGD)
      .with_learning_rate_boost_rate(1.0, 1.0);
  }

  inline const std::vector<unsigned>& topology() const { return _topology; }
  inline const activation::method& hidden_activation_method() const { return _hidden_activation; }
  inline const activation::method& output_activation_method() const { return _output_activation; }
  inline double learning_rate() const { return _learning_rate; }
  inline int number_of_epoch() const { return _number_of_epoch; }
  inline int batch_size() const { return _batch_size; }
  inline bool data_is_unique() const { return _data_is_unique; }
  inline const std::function<bool(NeuralNetworkHelper&)>& progress_callback() const { return _progress_callback; }
  inline const Logger& logger() const { return _logger; }
  inline int number_of_threads() const { return _number_of_threads; }
  inline double learning_rate_decay_rate() const { return _learning_rate_decay_rate; }
  inline bool adaptive_learning_rate() const { return _adaptive_learning_rate; }
  inline OptimiserType optimiser_type() const { return _optimiser_type; }
  inline double learning_rate_restart_rate() const { return _learning_rate_restart_rate; }
  inline double learning_rate_restart_boost() const { return _learning_rate_restart_boost; }

private:
  std::vector<unsigned> _topology;
  activation::method _hidden_activation;
  activation::method _output_activation;
  double _learning_rate;
  int _number_of_epoch;
  int _batch_size;
  bool _data_is_unique;
  std::function<bool(NeuralNetworkHelper&)> _progress_callback;
  Logger _logger;
  int _number_of_threads;
  double _learning_rate_decay_rate;
  bool _adaptive_learning_rate;
  OptimiserType _optimiser_type;
  double _learning_rate_restart_rate;
  double _learning_rate_restart_boost;
};

class NeuralNetworkHelper
{
public:
  class NeuralNetworkHelperMetrics
  {
  public:
    long double error() const { return _error; }
    long double forecast() const { return _forecast; }

    virtual ~NeuralNetworkHelperMetrics() = default;

  protected:
    friend class NeuralNetworkHelper;
    friend class NeuralNetwork;

    NeuralNetworkHelperMetrics(long double error, long double forecast) :
      _error(error),
      _forecast(forecast)
    {
    }
    NeuralNetworkHelperMetrics(const NeuralNetworkHelperMetrics& src)
    {
      *this = src;
    }
    NeuralNetworkHelperMetrics& operator=(const NeuralNetworkHelperMetrics& src)
    {
      if (this != &src)
      {
        _error = src._error;
        _forecast = src._forecast;
      }
      return *this;
    }
    NeuralNetworkHelperMetrics(NeuralNetworkHelperMetrics&& src) = delete;
    NeuralNetworkHelperMetrics& operator=(NeuralNetworkHelperMetrics&& src) = delete;

    long double _error;
    long double _forecast;
  };

  NeuralNetworkHelper() = delete;
  NeuralNetworkHelper(const NeuralNetworkHelper& src) noexcept
  {
    *this = src;
  }
  NeuralNetworkHelper(NeuralNetworkHelper&& src) noexcept
  {
    *this = std::move(src);
  }
  NeuralNetworkHelper& operator=(const NeuralNetworkHelper& src) noexcept
  {
    if (this != &src)
    {
      _neural_network = src._neural_network;
      _learning_rate = src._learning_rate;
      _number_of_epoch = src._number_of_epoch;
      _epoch = src._epoch;
      _training_indexes = src._training_indexes;
      _checking_indexes = src._checking_indexes;
      _final_check_indexes = src._final_check_indexes;
      _training_inputs = src._training_inputs;
      _training_outputs = src._training_outputs;
    }
    return *this;
  }

  NeuralNetworkHelper& operator=(NeuralNetworkHelper&& src) noexcept
  {
    if (this != &src)
    {
      _neural_network = src._neural_network;
      _learning_rate = src._learning_rate;
      _number_of_epoch = src._number_of_epoch;
      _epoch = src._epoch;
      _training_indexes = std::move(src._training_indexes);
      _checking_indexes = std::move(src._checking_indexes);
      _final_check_indexes = std::move(src._final_check_indexes);
      _training_inputs = std::move(src._training_inputs);
      _training_outputs = std::move(src._training_outputs);
      src._neural_network = nullptr;
      src._learning_rate = 0;
      src._number_of_epoch = 0;
      src._epoch = 0;
    }
    return *this;
  }
  virtual ~NeuralNetworkHelper() = default;

  double learning_rate() const { return _learning_rate; }
  void set_learning_rate(double learning_rate) { _learning_rate = learning_rate; }

  unsigned number_of_epoch() const { return _number_of_epoch; }
  unsigned epoch() const { return _epoch; }

  NeuralNetworkHelperMetrics get_metrics(NeuralNetworkOptions::ErrorCalculation error_type, NeuralNetworkOptions::ForecastAccuracy forecast_type) const;

protected:
  NeuralNetworkHelper(
    NeuralNetwork& neural_network,
    double learning_rate,
    unsigned number_of_epoch,
    const std::vector<std::vector<double>>& training_inputs,
    const std::vector<std::vector<double>>& training_outputs
  ) :
    _neural_network(&neural_network),
    _learning_rate(learning_rate),
    _number_of_epoch(number_of_epoch),
    _epoch(0),
    _training_inputs(training_inputs),
    _training_outputs(training_outputs)
  {
  }

  void set_epoch(unsigned epoch) { _epoch = epoch; }

  void move_indexes(
    std::vector<size_t>&& training_indexes,
    std::vector<size_t>&& checking_indexes,
    std::vector<size_t>&& final_check_indexes
  )
  {
    _training_indexes = training_indexes;
    _checking_indexes = checking_indexes;
    _final_check_indexes = final_check_indexes;
  }

  void move_training_indexes(std::vector<size_t>&& training_indexes)
  {
    _training_indexes = training_indexes;
  }

  const std::vector<size_t>& training_indexes() const { return _training_indexes; }
  const std::vector<size_t>& checking_indexes() const { return _checking_indexes; }
  const std::vector<size_t>& final_check_indexes() const { return _final_check_indexes; }

  const std::vector<std::vector<double>>& training_inputs() const { return _training_inputs; }
  const std::vector<std::vector<double>>& training_outputs() const { return _training_outputs; }

  friend class NeuralNetwork;

private:
  NeuralNetwork* _neural_network;
  double _learning_rate;
  unsigned _number_of_epoch;
  unsigned _epoch;
  std::vector<std::vector<double>> _training_inputs;
  std::vector<std::vector<double>> _training_outputs;
  std::vector<size_t> _training_indexes;
  std::vector<size_t> _checking_indexes;
  std::vector<size_t> _final_check_indexes;
};

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
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    }

    LayersAndNeuronsContainer(LayersAndNeuronsContainer&& src) noexcept :
      _offsets(std::move(src._offsets)),
      _total_size(src._total_size),
      _data(std::move(src._data)),
      _topology(std::move(src._topology))
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    }

    LayersAndNeuronsContainer& operator=(const LayersAndNeuronsContainer& src) noexcept
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
 
    LayersAndNeuronsContainer& operator=(LayersAndNeuronsContainer&& src) noexcept
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

    LayersAndNeuronsContainer(const std::vector<unsigned>& topology, bool shifted_by_one=false, bool add_bias=false) noexcept
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

    LayersAndNeuronsContainer& operator=(const std::vector<std::vector<double>>& data)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      assert(_topology.size() == data.size());
      for(size_t layer = 0; layer < data.size(); ++layer)
      {
        set(static_cast<unsigned>(layer), data[layer]);
      }
      return *this;
    }

    inline void zero()
    {
      std::fill(_data.begin(), _data.end(), 0.0);
    }

    inline void set( unsigned layer, unsigned neuron, double&& data)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      ensure_size(layer, neuron);
      _data[_offsets[layer]+neuron] = std::move(data);
    }

    inline void set( unsigned layer, unsigned neuron, const double& data)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      ensure_size(layer, neuron);
      _data[_offsets[layer]+neuron] = data;
    }
    void set(unsigned layer, const std::vector<double>& data)
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
    inline void add(const LayersAndNeuronsContainer& container)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      assert(_data.size() == container._data.size());
      for( size_t index = 0; index < _data.size(); ++index)
      {
        _data[index] += container._data[index];
      }
    }
    
    inline const double& get(unsigned layer, unsigned neuron) const
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      ensure_size(layer, neuron);
      return _data[_offsets[layer]+neuron];
    }

    std::vector<double> get_neurons(unsigned layer) const
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      std::vector<double> data;
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
    std::vector<double> _data;
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
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    }

    GradientsAndOutputs(const GradientsAndOutputs& src) noexcept:
      _outputs(src._outputs),
      _gradients(src._gradients),
      _batch_size(src._batch_size)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    }

    GradientsAndOutputs(GradientsAndOutputs&& src) noexcept: 
      _outputs(std::move(src._outputs)),
      _gradients(std::move(src._gradients)),
      _batch_size(src._batch_size)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      src._batch_size = 0;
    }

    GradientsAndOutputs& operator=(const GradientsAndOutputs& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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
      _batch_size = 0;
      _outputs.zero();
      _gradients.zero();
    }
    void add(const GradientsAndOutputs& src)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      _outputs.add(src._outputs);
      _gradients.add(src._gradients);
    }
    unsigned num_output_layers() const 
    { 
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      return static_cast<unsigned>(_outputs.number_layers());
    }

    unsigned batch_size() const
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      return _batch_size;
    }
    unsigned num_output_neurons(unsigned layer) const 
    { 
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      return static_cast<unsigned>(_outputs.number_neurons(layer));
    }

    double get_gradient(unsigned layer, unsigned neuron) const
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      return _gradients.get(layer, neuron);
    }

    void set_gradient(unsigned layer, unsigned neuron, double gradient)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      _gradients.set(layer, neuron, gradient);
    }

    void set_gradients(unsigned layer, const std::vector<double>& gradients)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      _gradients.set(layer, gradients);
    }

    void set_gradients(const LayersAndNeuronsContainer& gradients)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      _gradients = gradients;
    }

    void set_gradients(const std::vector<std::vector<double>>& gradients)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
      _gradients = gradients;
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
    LayersAndNeuronsContainer _outputs;
    LayersAndNeuronsContainer _gradients;
    unsigned _batch_size;
  };

public:
  NeuralNetwork(const NeuralNetworkOptions options);
  NeuralNetwork(const std::vector<unsigned>& topology, const activation::method& hidden_layer_activation, const activation::method& output_layer_activation, const Logger& logger);
  NeuralNetwork(const std::vector<Layer>& layers, const activation::method& hidden_layer_activation, const activation::method& output_layer_activation, const Logger& logger, long double error, long double mean_absolute_percentage_error);
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
  
  NeuralNetworkHelper::NeuralNetworkHelperMetrics get_metrics(NeuralNetworkOptions::ErrorCalculation error_type, NeuralNetworkOptions::ForecastAccuracy forecast_type) const;
  double get_learning_rate() const;

  NeuralNetworkOptions& options() { return _options;}
  const NeuralNetworkOptions& options() const { return _options;}

private:
  void calculate_back_propagation(
    GradientsAndOutputs& gradients,
    const std::vector<double>& outputs, 
    const std::vector<Layer>& layers) const;
  void calculate_forward_feed(
    GradientsAndOutputs& gradients,
    const std::vector<double>& inputs, 
    const std::vector<Layer>& layers) const;
  GradientsAndOutputs train_single_batch(
    const std::vector<std::vector<double>>::const_iterator inputs_begin, 
    const std::vector<std::vector<double>>::const_iterator outputs_begin,
    const size_t size
  ) const;

  std::vector<double> calculate_weight_gradients(unsigned layer_number, unsigned neuron_number, const GradientsAndOutputs& source) const;

  void apply_weight_gradients(std::vector<Layer>& layers, const std::vector<GradientsAndOutputs>& batch_activation_gradients, double learning_rate, unsigned epoch) const;
  void apply_weight_gradients(std::vector<Layer>& layers, const GradientsAndOutputs& batch_activation_gradient, double learning_rate, unsigned epoch) const;

  static std::vector<double> caclulate_output_gradients(const std::vector<double>& target_outputs, const std::vector<double>& given_outputs, const Layer& output_layer);

  NeuralNetworkHelper::NeuralNetworkHelperMetrics get_metrics(NeuralNetworkOptions::ErrorCalculation error_type, NeuralNetworkOptions::ForecastAccuracy forecast_type, bool final_check) const;

  double calculate_forecast_accuracy(NeuralNetworkOptions::ForecastAccuracy forecast_type, const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;

  // Calculates the Mean Absolute Percentage Error (MAPE)
  double calculate_forecast_accuracy_mape(const std::vector<double>& ground_truth, const std::vector<double>& predictions) const;
  double calculate_forecast_accuracy_mape(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;

  double calculate_forecast_accuracy_smape(const std::vector<double>& ground_truth, const std::vector<double>& predictions) const;
  double calculate_forecast_accuracy_smape(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const;

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
  void create_shuffled_indexes(NeuralNetworkHelper& neural_network_helper, size_t raw_size, bool data_is_unique) const;

  void log_training_info(
    const std::vector<std::vector<double>>& training_inputs,
    const std::vector<std::vector<double>>& training_outputs) const;

  std::vector<size_t> get_shuffled_indexes(size_t raw_size) const;

  double _learning_rate;
  std::vector<Layer> _layers;
  mutable std::shared_mutex _mutex;

  NeuralNetworkOptions _options;
  NeuralNetworkHelper* _neural_network_helper;
};