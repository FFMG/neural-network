#include "neuralnetworkhelper.h"
#include "neuralnetwork.h"

std::vector<NeuralNetworkHelperMetrics> NeuralNetworkHelper::calculate_forecast_metric(ErrorCalculation::type error_type) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
  return _neural_network->calculate_forecast_metric_all_layers(error_type);
}

std::vector<std::vector<NeuralNetworkHelperMetrics>> NeuralNetworkHelper::calculate_forecast_metrics(const std::vector<ErrorCalculation::type>& error_types) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
  return _neural_network->calculate_forecast_metrics_all_layers(error_types);
}
