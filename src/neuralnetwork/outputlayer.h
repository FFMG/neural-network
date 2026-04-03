#pragma once

#include "./libraries/instrumentor.h"

#include "outputlayerdetails.h"

class OutputLayer
{
public:
  OutputLayer(const std::vector<OutputLayerDetails>& output_layer_details) noexcept :
    _output_layer_details(output_layer_details)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
    create_activation_per_neuron(output_layer_details);
    create_using_activation_derivatives_per_neuron(output_layer_details);
  }

  OutputLayer(const OutputLayer& src) noexcept :
    _output_layer_details(src._output_layer_details),
    _activations(src._activations),
    _is_not_using_activation_derivatives(src._is_not_using_activation_derivatives)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
  }

  OutputLayer(OutputLayer&& src) noexcept :
    _output_layer_details(std::move(src._output_layer_details)),
    _activations(std::move(src._activations)),
    _is_not_using_activation_derivatives(std::move(src._is_not_using_activation_derivatives))
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
    }
    return *this;
  }

  virtual ~OutputLayer()
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayer");
  }

protected:
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
        const auto is_not_using_activation_derivative = Layer::is_not_using_activation_derivative(output_layer_detail.get_activation().get_method(), error_calculation_type);
        _is_not_using_activation_derivatives.push_back(is_not_using_activation_derivative);
      }
    }
  }

  std::vector<OutputLayerDetails> _output_layer_details;
  std::vector<activation> _activations;
  std::vector<uint8_t> _is_not_using_activation_derivatives;
};