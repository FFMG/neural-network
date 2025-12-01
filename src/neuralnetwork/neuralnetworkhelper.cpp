#include "neuralnetworkhelper.h"
#include "neuralnetwork.h"

NeuralNetworkHelper::NeuralNetworkHelperMetrics NeuralNetworkHelper::calculate_forecast_metric(ErrorCalculation::type error_type) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
  return _neural_network->calculate_forecast_metric(error_type);
}

std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> NeuralNetworkHelper::calculate_forecast_metrics(const std::vector<ErrorCalculation::type>& error_types) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
  return _neural_network->calculate_forecast_metrics(error_types);
}
