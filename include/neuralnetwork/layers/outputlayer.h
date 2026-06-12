#pragma once

#include "../libraries/instrumentor.h"

#include "../common/evaluationconfig.h"
#include "outputlayerdetails.h"


namespace myoddweb::nn
{
class OutputLayer
{
public:
  OutputLayer(const std::vector<OutputLayerDetails>& output_layer_details) noexcept :
    _output_layer_details(output_layer_details),
    _number_output_layers(static_cast<unsigned>(output_layer_details.size()))
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
    create_bounds(output_layer_details);
  }

  OutputLayer(const OutputLayer& src) noexcept :
    _output_layer_details(src._output_layer_details),
    _bounds(src._bounds),
    _number_output_layers(src._number_output_layers)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
  }

  OutputLayer(OutputLayer&& src) noexcept :
    _output_layer_details(std::move(src._output_layer_details)),
    _bounds(std::move(src._bounds)),
    _number_output_layers(src._number_output_layers)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
    src._number_output_layers = 0;
  }

  OutputLayer& operator=(const OutputLayer& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
    if (this != &src)
    {
      _output_layer_details = src._output_layer_details;
      _bounds = src._bounds;
      _number_output_layers = src._number_output_layers;
    }
    return *this;
  }

  OutputLayer& operator=(OutputLayer&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
    if (this != &src)
    {
      _output_layer_details = std::move(src._output_layer_details);
      _bounds = std::move(src._bounds);
      _number_output_layers = src._number_output_layers;
      src._number_output_layers = 0;
    }
    return *this;
  }

  virtual ~OutputLayer()
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
  }

  [[nodiscard]] inline const std::vector<OutputLayerDetails>& output_layer_details() const
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
    return _output_layer_details;
  }

protected:
  struct bounds {
    unsigned start;
    unsigned end;
  };
  
  [[nodiscard]] inline const bounds& layer_bounds(unsigned layer_number) const
  {
#if VALIDATE_DATA == 1
    if (layer_number >= _bounds.size())
    {
      Logger::panic("Trying to get bounds #", layer_number, " outsize the output layer size!");
    }
#endif
    return _bounds[layer_number];
  }

  [[nodiscard]] static inline bool is_not_using_activation_derivative(const activation::method method, const ErrorCalculation::type& error_calculation_type) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    switch (error_calculation_type)
    {
    case ErrorCalculation::type::bce_loss: return method == activation::method::sigmoid;
    case ErrorCalculation::type::cross_entropy: return method == activation::method::softmax;
    case ErrorCalculation::type::mse:
    case ErrorCalculation::type::rmse:
    case ErrorCalculation::type::huber_loss:
    case ErrorCalculation::type::log_cosh:
    case ErrorCalculation::type::none:
    case ErrorCalculation::type::huber_direction_loss:
    case ErrorCalculation::type::mae:
    case ErrorCalculation::type::nrmse:
    case ErrorCalculation::type::mape:
    case ErrorCalculation::type::smape:
    case ErrorCalculation::type::wape:
    case ErrorCalculation::type::directional_accuracy:
    case ErrorCalculation::type::directional_confidence_score:
    case ErrorCalculation::type::prediction_coverage:
      return false; // Always use derivative for MSE-like losses unless it's linear (where deriv=1 anyway)
    }
    return false;
  }

  [[nodiscard]] inline unsigned number_output_layers() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
    return _number_output_layers;
  }
  [[nodiscard]] inline const EvaluationConfig& evaluation_config(unsigned output_layer_index) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
#if VALIDATE_DATA == 1
    if (output_layer_index >= _output_layer_details.size())
    {
      Logger::panic("Trying to get EvaluationConfig past the number of output layers(#", output_layer_index,")!");
    }
#endif
    return _output_layer_details[output_layer_index].get_error_evaluation_config();
  }

private:
  void create_bounds(const std::vector<OutputLayerDetails>& output_layer_details)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
    _bounds.clear();
    _bounds.reserve(output_layer_details.size());

    unsigned start_neuron = 0;
    unsigned end_neuron = 0;
    for (const auto& output_layer_detail : output_layer_details)
    {
      end_neuron = start_neuron + output_layer_detail.get_size() - 1;
      _bounds.push_back({ start_neuron, end_neuron });
      start_neuron = end_neuron + 1;
    }
  }

  std::vector<OutputLayerDetails> _output_layer_details;
  std::vector<bounds> _bounds;
  unsigned _number_output_layers;
};
} // namespace myoddweb::nn
