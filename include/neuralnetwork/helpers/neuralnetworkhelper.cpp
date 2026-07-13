#include <algorithm>
#include "neuralnetworkhelper.h"
#include "../neuralnetwork.h"


namespace myoddweb::nn
{
NeuralNetworkHelper::NeuralNetworkHelper(
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
  _percent_complete(0.0),
  _training_inputs(&training_inputs),
  _training_outputs(&training_outputs),
  _training_indexes(std::make_shared<std::vector<size_t>>()),
  _checking_indexes(std::make_shared<std::vector<size_t>>()),
  _final_check_indexes(std::make_shared<std::vector<size_t>>()),
  _duration_ms(0.0),
  _last_epoch_time(),
  _max_history_size(std::clamp<size_t>(number_of_epoch / 2000, 10, 50)),
  _duration_sum(0.0),
  _ring_buffer_index(0)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
}

std::vector<NeuralNetworkHelperMetrics> NeuralNetworkHelper::calculate_forecast_metric(ErrorCalculation::type error_type) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
  return _neural_network->calculate_forecast_metric_all_layers(error_type);
}

std::vector<std::vector<NeuralNetworkHelperMetrics>> NeuralNetworkHelper::calculate_forecast_metrics(const std::vector<ErrorCalculation::type>& error_types, bool in_sample) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
  return _neural_network->calculate_forecast_metrics_all_layers(error_types, in_sample);
}

} // namespace myoddweb::nn
