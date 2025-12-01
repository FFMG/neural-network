#pragma once
#include <vector>

#include "./libraries/instrumentor.h"
#include "errorcalculation.h"
#include "neuralnetworkoptions.h"

class NeuralNetwork;
class NeuralNetworkHelper
{
public:
  class NeuralNetworkHelperMetrics
  {
  public:
    inline long double error() const noexcept { 
      MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
      return _error; 
    }
    inline ErrorCalculation::type error_type() const noexcept { 
      MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
      return _error_type; 
    }

    virtual ~NeuralNetworkHelperMetrics() = default;

    NeuralNetworkHelperMetrics(const NeuralNetworkHelperMetrics& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
      *this = src;
    }
    NeuralNetworkHelperMetrics& operator=(const NeuralNetworkHelperMetrics& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
      if (this != &src)
      {
        _error = src._error;
        _error_type = src._error_type;
      }
      return *this;
    }

    NeuralNetworkHelperMetrics(NeuralNetworkHelperMetrics&& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
      *this = src;
    }
    NeuralNetworkHelperMetrics& operator=(NeuralNetworkHelperMetrics&& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
      if (this != &src)
      {
        _error = src._error;
        _error_type = src._error_type;
        src._error = 0.0;
        src._error_type = ErrorCalculation::type::none;
      }
      return *this;
    }

  protected:
    friend class NeuralNetworkHelper;
    friend class NeuralNetwork;

    NeuralNetworkHelperMetrics(long double error, ErrorCalculation::type error_type) noexcept :
      _error(error),
      _error_type(error_type)
    {
      MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
    }

    long double _error;
    ErrorCalculation::type _error_type;
  };

  NeuralNetworkHelper() = delete;
  NeuralNetworkHelper(const NeuralNetworkHelper& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    *this = src;
  }
  NeuralNetworkHelper(NeuralNetworkHelper&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    *this = std::move(src);
  }
  NeuralNetworkHelper& operator=(const NeuralNetworkHelper& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
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
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
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

  inline double learning_rate() const noexcept
  { 
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return _learning_rate; 
  }
  
  void set_learning_rate(double learning_rate) noexcept {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    _learning_rate = learning_rate; 
  }

  inline unsigned number_of_epoch() const noexcept
  { 
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return _number_of_epoch; 
  }
  inline unsigned epoch() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return _epoch; 
  }

  inline size_t sample_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
    return _training_inputs.size();
  }

  NeuralNetworkHelperMetrics calculate_forecast_metric(ErrorCalculation::type error_type) const;
  std::vector<NeuralNetworkHelperMetrics> calculate_forecast_metrics(const std::vector<ErrorCalculation::type>& error_types) const;

  NeuralNetworkHelper(
    NeuralNetwork& neural_network,
    double learning_rate,
    unsigned number_of_epoch,
    const std::vector<std::vector<double>>& training_inputs,
    const std::vector<std::vector<double>>& training_outputs
  ) noexcept :
    _neural_network(&neural_network),
    _learning_rate(learning_rate),
    _number_of_epoch(number_of_epoch),
    _epoch(0),
    _training_inputs(training_inputs),
    _training_outputs(training_outputs)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
  }

  void set_epoch(unsigned epoch) noexcept {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    _epoch = epoch; 
  }

  void move_indexes (
    std::vector<size_t>&& training_indexes,
    std::vector<size_t>&& checking_indexes,
    std::vector<size_t>&& final_check_indexes
  )
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    _training_indexes = std::move(training_indexes);
    _checking_indexes = std::move(checking_indexes);
    _final_check_indexes = std::move(final_check_indexes);
  }

  void move_training_indexes(std::vector<size_t>&& training_indexes)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    _training_indexes = std::move(training_indexes);
  }

  const std::vector<size_t>& training_indexes() const noexcept { 
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return _training_indexes; 
  }
  const std::vector<size_t>& checking_indexes() const noexcept { 
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return _checking_indexes; 
  }
  const std::vector<size_t>& final_check_indexes() const noexcept {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return _final_check_indexes; 
  }
  const std::vector<std::vector<double>>& training_inputs() const noexcept {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return _training_inputs; 
  }
  const std::vector<std::vector<double>>& training_outputs() const noexcept {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return _training_outputs; 
  }

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
