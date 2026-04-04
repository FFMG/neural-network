#pragma once

#include "./libraries/instrumentor.h"

#include "layer.h"
#include "logger.h"
#include "outputlayerdetails.h"

class OutputLayer
{
public:
  OutputLayer(const std::vector<OutputLayerDetails>& output_layer_details) noexcept :
    _output_layer_details(output_layer_details),
    _number_output_layers(static_cast<unsigned>(output_layer_details.size()))
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
    create_activation_per_neuron(output_layer_details);
    create_using_activation_derivatives_per_neuron(output_layer_details);
    create_bounds(output_layer_details);
  }

  OutputLayer(const OutputLayer& src) noexcept :
    _output_layer_details(src._output_layer_details),
    _activations(src._activations),
    _is_not_using_activation_derivatives(src._is_not_using_activation_derivatives),
    _bounds(src._bounds),
    _number_output_layers(src._number_output_layers)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
  }

  OutputLayer(OutputLayer&& src) noexcept :
    _output_layer_details(std::move(src._output_layer_details)),
    _activations(std::move(src._activations)),
    _is_not_using_activation_derivatives(std::move(src._is_not_using_activation_derivatives)),
    _bounds(std::move(src._bounds)),
    _number_output_layers(src._number_output_layers)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
  }

  OutputLayer& operator=(const OutputLayer& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
    if (this != &src)
    {
      _output_layer_details = src._output_layer_details;
      _activations = src._activations;
      _is_not_using_activation_derivatives = src._is_not_using_activation_derivatives;
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
      _activations = std::move(src._activations);
      _is_not_using_activation_derivatives = std::move(src._is_not_using_activation_derivatives);
      _bounds = std::move(src._bounds);
      _number_output_layers = src._number_output_layers;
    }
    return *this;
  }

  virtual ~OutputLayer()
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
  }

protected:
  struct bounds {
    unsigned start;
    unsigned end;
  };
  
  [[nodiscard]] inline const bounds& layer_bounds(unsigned layer_number) const
  {
#if VALIDATE_DATA ==1
    if (layer_number >= _bounds.size())
    {
      Logger::panic("Trying to get bounds #", layer_number, " outsize the output layer size!");
    }
#endif
    return _bounds[layer_number];
  }

  [[nodiscard]] inline const std::vector<OutputLayerDetails>& output_layer_details() const
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
    return _output_layer_details;
  }

  [[nodiscard]] inline const activation& get_activation(unsigned neuron_index) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
#if VALIDATE_DATA == 1
    if (neuron_index >= _activations.size())
    {
      Logger::panic("Trying to get an activation layer outside of the index!");
    }
#endif
    return _activations[neuron_index];
  }

  [[nodiscard]] inline bool get_is_not_using_activation_derivatives(unsigned neuron_index) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
#if VALIDATE_DATA == 1
    if (neuron_index >= _is_not_using_activation_derivatives.size())
    {
      Logger::panic("Trying to get if using an activation derivative outside of the index!");
    }
#endif
    return _is_not_using_activation_derivatives[neuron_index] != 0;
  }

  [[nodiscard]] static inline bool is_not_using_activation_derivative(const activation::method method, const ErrorCalculation::type& error_calculation_type) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    switch (error_calculation_type)
    {
    case ErrorCalculation::type::bce_loss: return method == activation::method::sigmoid;
    case ErrorCalculation::type::cross_entropy: return method == activation::method::softmax;
    case ErrorCalculation::type::mse:
    case ErrorCalculation::type::huber_loss:
    case ErrorCalculation::type::log_cosh: return method == activation::method::linear;
    }
    return false;
  }

  [[nodiscard]] inline unsigned number_output_layers() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
    return _number_output_layers;
  }
  [[nodiscard]] inline const ErrorCalculation::EvaluationConfig& evaluation_config(unsigned output_layer_index) const noexcept
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
  void create_activation_per_neuron(const std::vector<OutputLayerDetails>& output_layer_details)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
    for (const auto& output_layer_detail : output_layer_details)
    {
      for (size_t i = 0; i < output_layer_detail.get_size(); ++i)
      {
        _activations.push_back(output_layer_detail.get_activation());
      }
    }
  }

  void create_using_activation_derivatives_per_neuron(const std::vector<OutputLayerDetails>& output_layer_details)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
    for (const auto& output_layer_detail : output_layer_details)
    {
      for (size_t i = 0; i < output_layer_detail.get_size(); ++i)
      {
        const auto& error_calculation_type = output_layer_detail.get_output_error_calculation_type();
        const auto is_not = is_not_using_activation_derivative(output_layer_detail.get_activation().get_method(), error_calculation_type);
        _is_not_using_activation_derivatives.push_back(is_not);
      }
    }
  }

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
  std::vector<activation> _activations;
  std::vector<uint8_t> _is_not_using_activation_derivatives;
  std::vector<bounds> _bounds;
  unsigned _number_output_layers;
};