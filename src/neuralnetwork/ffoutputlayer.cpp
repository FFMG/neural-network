#include "./libraries/instrumentor.h"
#include "ffoutputlayer.h"
#include "logger.h"
#include <numeric>

constexpr bool _has_bias_neuron = true;

FFOutputLayer::FFOutputLayer(
  unsigned layer_index,
  const OutputLayerDetails& output_layer_detail,
  unsigned num_neurons_in_previous_layer,
  unsigned num_neurons_in_this_layer,
  double weight_decay,
  const OptimiserType& optimiser_type,
  int residual_layer_number,
  ResidualProjector* residual_projector,
  int number_of_threads
) :
  FFLayer(
    layer_index,
    num_neurons_in_previous_layer,
    num_neurons_in_this_layer,
    weight_decay,
    Layer::LayerType::Output,
    output_layer_detail.get_activation(),
    optimiser_type,
    residual_layer_number,
    0.0, // no dropout for output layer
    residual_projector,
    number_of_threads),
  _output_layer_detail(output_layer_detail)
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
}

FFOutputLayer::FFOutputLayer(const FFOutputLayer& src) noexcept :
  FFLayer(src),
  _output_layer_detail( src._output_layer_detail)
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
}

FFOutputLayer::FFOutputLayer(
  unsigned layer_index,
  const OutputLayerDetails& output_layer_detail,
  const OptimiserType optimiser_type,
  int residual_layer_number,
  unsigned number_input_neurons,
  unsigned number_output_neurons,
  const std::vector<Neuron>& neurons,
  const std::vector<double>& w_values,
  const std::vector<double>& w_grads,
  const std::vector<double>& w_velocities,
  const std::vector<double>& w_m1,
  const std::vector<double>& w_m2,
  const std::vector<long long>& w_timesteps,
  const std::vector<double>& w_decays,
  const std::vector<double>& b_values,
  const std::vector<double>& b_grads,
  const std::vector<double>& b_velocities,
  const std::vector<double>& b_m1,
  const std::vector<double>& b_m2,
  const std::vector<long long>& b_timesteps,
  const std::vector<double>& b_decays,
  const ResidualProjector* residual_projector,
  int number_of_threads
) noexcept : 
  FFLayer(
  layer_index,
  Layer::LayerType::Output,
  output_layer_detail.get_activation(), 
  optimiser_type,
  residual_layer_number,
  number_input_neurons,
  number_output_neurons,
  neurons,
  w_values,
  w_grads,
  w_velocities,
  w_m1,
  w_m2,
  w_timesteps,
  w_decays,
  b_values,
  b_grads,
  b_velocities,
  b_m1,
  b_m2,
  b_timesteps,
  b_decays,
  residual_projector,
  number_of_threads),
  _output_layer_detail(output_layer_detail)
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
}

FFOutputLayer::FFOutputLayer(FFOutputLayer&& src) noexcept :
  FFLayer(std::move(src)),
  _output_layer_detail(std::move(src._output_layer_detail))
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
}

FFOutputLayer& FFOutputLayer::operator=(const FFOutputLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  if(this != &src)
  {
    FFLayer::operator=(src);
    _output_layer_detail = src._output_layer_detail;
  }
  return *this;
}

FFOutputLayer& FFOutputLayer::operator=(FFOutputLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  if(this != &src)
  {
    FFLayer::operator=(std::move(src));
    _output_layer_detail = std::move(src._output_layer_detail);
  }
  return *this;
}

FFOutputLayer::~FFOutputLayer()
{
}

bool FFOutputLayer::has_bias() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  return _has_bias_neuron;
}

Layer* FFOutputLayer::clone() const
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  return new FFOutputLayer(*this);
}

void FFOutputLayer::calculate_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states) const
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  const auto& error_calculation_type = _output_layer_detail.get_output_error_calculation_type();
  const auto& evaluation_config = _output_layer_detail.get_error_evaluation_config();
  const size_t batch_size = batch_gradients_and_outputs.size();
  const size_t N_total = get_number_neurons();
  const auto is_not_using_activation_derivative = Layer::is_not_using_activation_derivative(error_calculation_type);

  auto run_output_gradients = [&](size_t start, size_t end)
    {
      std::vector<double> gradients(N_total, 0.0);
      std::vector<double> deltas(N_total, 0.0);

      for (size_t b = start; b < end; b++)
      {
        const auto& given_outputs = batch_gradients_and_outputs[b].get_outputs(get_layer_index());
        const auto& target_outputs = *(target_outputs_begin + b);

        calculate_error_deltas(deltas, target_outputs, given_outputs, error_calculation_type, evaluation_config, 0, N_total -1);

        if (is_not_using_activation_derivative)
        {
          for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
          {
            gradients[neuron_index] = deltas[neuron_index];
          }
        }
        else
        {
          const auto& current_hidden_state = batch_hidden_states[b].at(get_layer_index())[0];
          for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
          {
            double deriv = get_activation().activate_derivative(current_hidden_state.get_pre_activation_sum_at_neuron(neuron_index));
            gradients[neuron_index] = deltas[neuron_index] * deriv;
          }
        }
        batch_gradients_and_outputs[b].set_gradients(get_layer_index(), gradients);
      }
    };

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads <= 1)
  {
    run_output_gradients(0, batch_size);
  }
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t size = (batch_size / num_threads) + (t < (batch_size % num_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([=]()
          {
            run_output_gradients(start, end);
          });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}