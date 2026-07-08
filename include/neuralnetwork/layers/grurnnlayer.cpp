#include "../libraries/instrumentor.h"
#include "grurnnlayer.h"
#include "fflayer.h"
#include "../common/logger.h"
#include "../common/simd_utils.h"
#include <numeric>


namespace myoddweb::nn
{
GRURNNLayer::GRURNNLayer(
  unsigned layer_index,
  unsigned num_neurons_in_previous_layer, 
  unsigned num_neurons_in_this_layer, 
  double weight_decay,
  const Role layer_role,
  const activation& activation_method,
  const OptimiserType& optimiser_type, 
  int residual_layer_number,
  double dropout_rate,
  ResidualProjector* residual_projector,
  int number_of_threads,
  bool has_bias,
  double momentum
  ) :
  GRURNNLayer(
    layer_index,
    num_neurons_in_previous_layer,
    num_neurons_in_this_layer,
    std::vector<double>(static_cast<size_t>(num_neurons_in_previous_layer) * num_neurons_in_this_layer, weight_decay),
    layer_role,
    activation_method,
    optimiser_type,
    residual_layer_number,
    dropout_rate,
    residual_projector,
    number_of_threads,
    has_bias,
    momentum
  )
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");

  allocate_workspace();
}

GRURNNLayer::GRURNNLayer(
  unsigned layer_index,
  unsigned num_neurons_in_previous_layer,
  unsigned num_neurons_in_this_layer,
  const std::vector<double>& weight_decays,
  const Role layer_role,
  const activation& activation_method,
  const OptimiserType& optimiser_type,
  int residual_layer_number,
  double dropout_rate,
  ResidualProjector* residual_projector,
  int number_of_threads,
  bool has_bias,
  double momentum
) :
  Layer(
    layer_index,
    layer_role,
    activation_method,
    optimiser_type,
    residual_layer_number,
    num_neurons_in_previous_layer,
    num_neurons_in_this_layer,
    create_neurons(dropout_rate, num_neurons_in_this_layer),
    has_bias,
    weight_decays,
    residual_projector,
    number_of_threads,
    momentum
  )
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  
  initialize_recurrent_weights(weight_decays.empty() ? 0.0 : weight_decays[0]);

  allocate_workspace();
}

GRURNNLayer::GRURNNLayer(const GRURNNLayer& src) noexcept :
  Layer(src),
  _rw_values(src._rw_values),
  _rw_grads(src._rw_grads),
  _rw_velocities(src._rw_velocities),
  _rw_m1(src._rw_m1),
  _rw_m2(src._rw_m2),
  _rw_timesteps(src._rw_timesteps),
  _rw_decays(src._rw_decays),
  // Update Gate (z)
  _z_w_values(src._z_w_values),
  _z_w_grads(src._z_w_grads),
  _z_w_velocities(src._z_w_velocities),
  _z_w_m1(src._z_w_m1),
  _z_w_m2(src._z_w_m2),
  _z_w_timesteps(src._z_w_timesteps),
  _z_w_decays(src._z_w_decays),
  _z_rw_values(src._z_rw_values),
  _z_rw_grads(src._z_rw_grads),
  _z_rw_velocities(src._z_rw_velocities),
  _z_rw_m1(src._z_rw_m1),
  _z_rw_m2(src._z_rw_m2),
  _z_rw_timesteps(src._z_rw_timesteps),
  _z_rw_decays(src._z_rw_decays),
  _z_b_values(src._z_b_values),
  _z_b_grads(src._z_b_grads),
  _z_b_velocities(src._z_b_velocities),
  _z_b_m1(src._z_b_m1),
  _z_b_m2(src._z_b_m2),
  _z_b_timesteps(src._z_b_timesteps),
  _z_b_decays(src._z_b_decays),
  // Reset Gate (r)
  _r_w_values(src._r_w_values),
  _r_w_grads(src._r_w_grads),
  _r_w_velocities(src._r_w_velocities),
  _r_w_m1(src._r_w_m1),
  _r_w_m2(src._r_w_m2),
  _r_w_timesteps(src._r_w_timesteps),
  _r_w_decays(src._r_w_decays),
  _r_rw_values(src._r_rw_values),
  _r_rw_grads(src._r_rw_grads),
  _r_rw_velocities(src._r_rw_velocities),
  _r_rw_m1(src._r_rw_m1),
  _r_rw_m2(src._r_rw_m2),
  _r_rw_timesteps(src._r_rw_timesteps),
  _r_rw_decays(src._r_rw_decays),
  _r_b_values(src._r_b_values),
  _r_b_grads(src._r_b_grads),
  _r_b_velocities(src._r_b_velocities),
  _r_b_m1(src._r_b_m1),
  _r_b_m2(src._r_b_m2),
  _r_b_timesteps(src._r_b_timesteps),
  _r_b_decays(src._r_b_decays)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  _identity_proxy = nullptr;
  cache_recurrent_weights();
  allocate_workspace();
}

GRURNNLayer::GRURNNLayer(GRURNNLayer&& src) noexcept :
  Layer(std::move(src)),
  _rw_values(std::move(src._rw_values)),
  _rw_grads(std::move(src._rw_grads)),
  _rw_velocities(std::move(src._rw_velocities)),
  _rw_m1(std::move(src._rw_m1)),
  _rw_m2(std::move(src._rw_m2)),
  _rw_timesteps(std::move(src._rw_timesteps)),
  _rw_decays(std::move(src._rw_decays)),
  // Update Gate (z)
  _z_w_values(std::move(src._z_w_values)),
  _z_w_grads(std::move(src._z_w_grads)),
  _z_w_velocities(std::move(src._z_w_velocities)),
  _z_w_m1(std::move(src._z_w_m1)),
  _z_w_m2(std::move(src._z_w_m2)),
  _z_w_timesteps(std::move(src._z_w_timesteps)),
  _z_w_decays(std::move(src._z_w_decays)),
  _z_rw_values(std::move(src._z_rw_values)),
  _z_rw_grads(std::move(src._z_rw_grads)),
  _z_rw_velocities(std::move(src._z_rw_velocities)),
  _z_rw_m1(std::move(src._z_rw_m1)),
  _z_rw_m2(std::move(src._z_rw_m2)),
  _z_rw_timesteps(std::move(src._z_rw_timesteps)),
  _z_rw_decays(std::move(src._z_rw_decays)),
  _z_b_values(std::move(src._z_b_values)),
  _z_b_grads(std::move(src._z_b_grads)),
  _z_b_velocities(std::move(src._z_b_velocities)),
  _z_b_m1(std::move(src._z_b_m1)),
  _z_b_m2(std::move(src._z_b_m2)),
  _z_b_timesteps(std::move(src._z_b_timesteps)),
  _z_b_decays(std::move(src._z_b_decays)),
  // Reset Gate (r)
  _r_w_values(std::move(src._r_w_values)),
  _r_w_grads(std::move(src._r_w_grads)),
  _r_w_velocities(std::move(src._r_w_velocities)),
  _r_w_m1(std::move(src._r_w_m1)),
  _r_w_m2(std::move(src._r_w_m2)),
  _r_w_timesteps(std::move(src._r_w_timesteps)),
  _r_w_decays(std::move(src._r_w_decays)),
  _r_rw_values(std::move(src._r_rw_values)),
  _r_rw_grads(std::move(src._r_rw_grads)),
  _r_rw_velocities(std::move(src._r_rw_velocities)),
  _r_rw_m1(std::move(src._r_rw_m1)),
  _r_rw_m2(std::move(src._r_rw_m2)),
  _r_rw_timesteps(std::move(src._r_rw_timesteps)),
  _r_rw_decays(std::move(src._r_rw_decays)),
  _r_b_values(std::move(src._r_b_values)),
  _r_b_grads(std::move(src._r_b_grads)),
  _r_b_velocities(std::move(src._r_b_velocities)),
  _r_b_m1(std::move(src._r_b_m1)),
  _r_b_m2(std::move(src._r_b_m2)),
  _r_b_timesteps(std::move(src._r_b_timesteps)),
  _r_b_decays(std::move(src._r_b_decays)),
  _rw_values_T(std::move(src._rw_values_T)),
  _z_rw_values_T(std::move(src._z_rw_values_T)),
  _r_rw_values_T(std::move(src._r_rw_values_T)),
  _w_values_T(std::move(src._w_values_T)),
  _z_w_values_T(std::move(src._z_w_values_T)),
  _r_w_values_T(std::move(src._r_w_values_T)),
  _thread_workspaces(std::move(src._thread_workspaces))
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  _identity_proxy = src._identity_proxy;
  src._identity_proxy = nullptr;
}

GRURNNLayer::GRURNNLayer(
  unsigned layer_index,
  const Role layer_role,
  const OptimiserType optimiser_type,
  int residual_layer_number,
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
  const std::vector<double>& rw_values,
  const std::vector<double>& rw_grads,
  const std::vector<double>& rw_velocities,
  const std::vector<double>& rw_m1,
  const std::vector<double>& rw_m2,
  const std::vector<long long>& rw_timesteps,
  const std::vector<double>& rw_decays,
  // Update Gate (z)
  const std::vector<double>& z_w_values,
  const std::vector<double>& z_w_grads,
  const std::vector<double>& z_w_velocities,
  const std::vector<double>& z_w_m1,
  const std::vector<double>& z_w_m2,
  const std::vector<long long>& z_w_timesteps,
  const std::vector<double>& z_w_decays,
  const std::vector<double>& z_rw_values,
  const std::vector<double>& z_rw_grads,
  const std::vector<double>& z_rw_velocities,
  const std::vector<double>& z_rw_m1,
  const std::vector<double>& z_rw_m2,
  const std::vector<long long>& z_rw_timesteps,
  const std::vector<double>& z_rw_decays,
  const std::vector<double>& z_b_values,
  const std::vector<double>& z_b_grads,
  const std::vector<double>& z_b_velocities,
  const std::vector<double>& z_b_m1,
  const std::vector<double>& z_b_m2,
  const std::vector<long long>& z_b_timesteps,
  const std::vector<double>& z_b_decays,
  // Reset Gate (r)
  const std::vector<double>& r_w_values,
  const std::vector<double>& r_w_grads,
  const std::vector<double>& r_w_velocities,
  const std::vector<double>& r_w_m1,
  const std::vector<double>& r_w_m2,
  const std::vector<long long>& r_w_timesteps,
  const std::vector<double>& r_w_decays,
  const std::vector<double>& r_rw_values,
  const std::vector<double>& r_rw_grads,
  const std::vector<double>& r_rw_velocities,
  const std::vector<double>& r_rw_m1,
  const std::vector<double>& r_rw_m2,
  const std::vector<long long>& r_rw_timesteps,
  const std::vector<double>& r_rw_decays,
  const std::vector<double>& r_b_values,
  const std::vector<double>& r_b_grads,
  const std::vector<double>& r_b_velocities,
  const std::vector<double>& r_b_m1,
  const std::vector<double>& r_b_m2,
  const std::vector<long long>& r_b_timesteps,
  const std::vector<double>& r_b_decays,
  const ResidualProjector* residual_projector,
  int number_of_threads,
  const layer_activation_helper& lah,
  double momentum
) noexcept :
  Layer(
    layer_index,
    layer_role,
    optimiser_type,
    residual_layer_number,
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
    number_of_threads,
    lah, 
    momentum),
    _rw_values(rw_values),
    _rw_grads(rw_grads),
    _rw_velocities(rw_velocities),
    _rw_m1(rw_m1),
    _rw_m2(rw_m2),
    _rw_timesteps(rw_timesteps),
    _rw_decays(rw_decays),
    // Update Gate (z)
    _z_w_values(z_w_values),
    _z_w_grads(z_w_grads),
    _z_w_velocities(z_w_velocities),
    _z_w_m1(z_w_m1),
    _z_w_m2(z_w_m2),
    _z_w_timesteps(z_w_timesteps),
    _z_w_decays(z_w_decays),
    _z_rw_values(z_rw_values),
    _z_rw_grads(z_rw_grads),
    _z_rw_velocities(z_rw_velocities),
    _z_rw_m1(z_rw_m1),
    _z_rw_m2(z_rw_m2),
    _z_rw_timesteps(z_rw_timesteps),
    _z_rw_decays(z_rw_decays),
    _z_b_values(z_b_values),
    _z_b_grads(z_b_grads),
    _z_b_velocities(z_b_velocities),
    _z_b_m1(z_b_m1),
    _z_b_m2(z_b_m2),
    _z_b_timesteps(z_b_timesteps),
    _z_b_decays(z_b_decays),
    // Reset Gate (r)
    _r_w_values(r_w_values),
    _r_w_grads(r_w_grads),
    _r_w_velocities(r_w_velocities),
    _r_w_m1(r_w_m1),
    _r_w_m2(r_w_m2),
    _r_w_timesteps(r_w_timesteps),
    _r_w_decays(r_w_decays),
    _r_rw_values(r_rw_values),
    _r_rw_grads(r_rw_grads),
    _r_rw_velocities(r_rw_velocities),
    _r_rw_m1(r_rw_m1),
    _r_rw_m2(r_rw_m2),
    _r_rw_timesteps(r_rw_timesteps),
    _r_rw_decays(r_rw_decays),
    _r_b_values(r_b_values),
    _r_b_grads(r_b_grads),
    _r_b_velocities(r_b_velocities),
    _r_b_m1(r_b_m1),
    _r_b_m2(r_b_m2),
    _r_b_timesteps(r_b_timesteps),
    _r_b_decays(r_b_decays)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  cache_recurrent_weights();
  allocate_workspace();
}

GRURNNLayer& GRURNNLayer::operator=(const GRURNNLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  if(this != &src)
  {
    Layer::operator=(src);
    _rw_values = src._rw_values;
    _rw_grads = src._rw_grads;
    _rw_velocities = src._rw_velocities;
    _rw_m1 = src._rw_m1;
    _rw_m2 = src._rw_m2;
    _rw_timesteps = src._rw_timesteps;
    _rw_decays = src._rw_decays;

    // Update Gate (z)
    _z_w_values = src._z_w_values;
    _z_w_grads = src._z_w_grads;
    _z_w_velocities = src._z_w_velocities;
    _z_w_m1 = src._z_w_m1;
    _z_w_m2 = src._z_w_m2;
    _z_w_timesteps = src._z_w_timesteps;
    _z_w_decays = src._z_w_decays;
    _z_rw_values = src._z_rw_values;
    _z_rw_grads = src._z_rw_grads;
    _z_rw_velocities = src._z_rw_velocities;
    _z_rw_m1 = src._z_rw_m1;
    _z_rw_m2 = src._z_rw_m2;
    _z_rw_timesteps = src._z_rw_timesteps;
    _z_rw_decays = src._z_rw_decays;
    _z_b_values = src._z_b_values;
    _z_b_grads = src._z_b_grads;
    _z_b_velocities = src._z_b_velocities;
    _z_b_m1 = src._z_b_m1;
    _z_b_m2 = src._z_b_m2;
    _z_b_timesteps = src._z_b_timesteps;
    _z_b_decays = src._z_b_decays;

    // Reset Gate (r)
    _r_w_values = src._r_w_values;
    _r_w_grads = src._r_w_grads;
    _r_w_velocities = src._r_w_velocities;
    _r_w_m1 = src._r_w_m1;
    _r_w_m2 = src._r_w_m2;
    _r_w_timesteps = src._r_w_timesteps;
    _r_w_decays = src._r_w_decays;
    _r_rw_values = src._r_rw_values;
    _r_rw_grads = src._r_rw_grads;
    _r_rw_velocities = src._r_rw_velocities;
    _r_rw_m1 = src._r_rw_m1;
    _r_rw_m2 = src._r_rw_m2;
    _r_rw_timesteps = src._r_rw_timesteps;
    _r_rw_decays = src._r_rw_decays;
    _r_b_values = src._r_b_values;
    _r_b_grads = src._r_b_grads;
    _r_b_velocities = src._r_b_velocities;
    _r_b_m1 = src._r_b_m1;
    _r_b_m2 = src._r_b_m2;
    _r_b_timesteps = src._r_b_timesteps;
    _r_b_decays = src._r_b_decays;
    delete _identity_proxy;
    _identity_proxy = nullptr;
    allocate_workspace();
    cache_recurrent_weights();
  }
  return *this;
}

GRURNNLayer& GRURNNLayer::operator=(GRURNNLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  if(this != &src)
  {
    Layer::operator=(std::move(src));
    _rw_values = std::move(src._rw_values);
    _rw_grads = std::move(src._rw_grads);
    _rw_velocities = std::move(src._rw_velocities);
    _rw_m1 = std::move(src._rw_m1);
    _rw_m2 = std::move(src._rw_m2);
    _rw_timesteps = std::move(src._rw_timesteps);
    _rw_decays = std::move(src._rw_decays);

    // Update Gate (z)
    _z_w_values = std::move(src._z_w_values);
    _z_w_grads = std::move(src._z_w_grads);
    _z_w_velocities = std::move(src._z_w_velocities);
    _z_w_m1 = std::move(src._z_w_m1);
    _z_w_m2 = std::move(src._z_w_m2);
    _z_w_timesteps = std::move(src._z_w_timesteps);
    _z_w_decays = std::move(src._z_w_decays);
    _z_rw_values = std::move(src._z_rw_values);
    _z_rw_grads = std::move(src._z_rw_grads);
    _z_rw_velocities = std::move(src._z_rw_velocities);
    _z_rw_m1 = std::move(src._z_rw_m1);
    _z_rw_m2 = std::move(src._z_rw_m2);
    _z_rw_timesteps = std::move(src._z_rw_timesteps);
    _z_rw_decays = std::move(src._z_rw_decays);
    _z_b_values = std::move(src._z_b_values);
    _z_b_grads = std::move(src._z_b_grads);
    _z_b_velocities = std::move(src._z_b_velocities);
    _z_b_m1 = std::move(src._z_b_m1);
    _z_b_m2 = std::move(src._z_b_m2);
    _z_b_timesteps = std::move(src._z_b_timesteps);
    _z_b_decays = std::move(src._z_b_decays);
    // Reset Gate (r)
    _r_w_values = std::move(src._r_w_values);
    _r_w_grads = std::move(src._r_w_grads);
    _r_w_velocities = std::move(src._r_w_velocities);
    _r_w_m1 = std::move(src._r_w_m1);
    _r_w_m2 = std::move(src._r_w_m2);
    _r_w_timesteps = std::move(src._r_w_timesteps);
    _r_w_decays = std::move(src._r_w_decays);
    _r_rw_values = std::move(src._r_rw_values);
    _r_rw_grads = std::move(src._r_rw_grads);
    _r_rw_velocities = std::move(src._r_rw_velocities);
    _r_rw_m1 = std::move(src._r_rw_m1);
    _r_rw_m2 = std::move(src._r_rw_m2);
    _r_rw_timesteps = std::move(src._r_rw_timesteps);
    _r_rw_decays = std::move(src._r_rw_decays);
    _r_b_values = std::move(src._r_b_values);
    _r_b_grads = std::move(src._r_b_grads);
    _r_b_velocities = std::move(src._r_b_velocities);
    _r_b_m1 = std::move(src._r_b_m1);
    _r_b_m2 = std::move(src._r_b_m2);
    _r_b_timesteps = std::move(src._r_b_timesteps);
    _r_b_decays = std::move(src._r_b_decays);
    _rw_values_T = std::move(src._rw_values_T);
    _z_rw_values_T = std::move(src._z_rw_values_T);
    delete _identity_proxy;
    _identity_proxy = src._identity_proxy;
    src._identity_proxy = nullptr;
    _r_rw_values_T = std::move(src._r_rw_values_T);
    _w_values_T = std::move(src._w_values_T);
    _z_w_values_T = std::move(src._z_w_values_T);
    _r_w_values_T = std::move(src._r_w_values_T);
    _thread_workspaces = std::move(src._thread_workspaces);
  }
  return *this;
}

GRURNNLayer::~GRURNNLayer()
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  delete _identity_proxy;
}

void GRURNNLayer::init_bias(
  std::vector<double>& values, 
  std::vector<double>& grads,
  std::vector<double>& velocities, 
  std::vector<double>& m1,
  std::vector<double>& m2, 
  std::vector<long long>& timesteps,
  std::vector<double>& decays) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const auto num_neurons = get_number_neurons();
  values.assign(num_neurons, 0.0);
  grads.assign(num_neurons, 0.0);
  velocities.assign(num_neurons, 0.0);
  m1.assign(num_neurons, 0.0);
  m2.assign(num_neurons, 0.0);
  timesteps.assign(num_neurons, 0);
  decays.assign(num_neurons, 0.0); // No decay for biases
}

void GRURNNLayer::init_weights(
  std::vector<double>& values, std::vector<double>& grads,
  std::vector<double>& velocities, std::vector<double>& m1,
  std::vector<double>& m2, std::vector<long long>& timesteps,
  std::vector<double>& decays, size_t size, bool is_input,
  double weight_decay) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const auto num_neurons = get_number_output_neurons();
  const auto num_inputs = get_number_input_neurons();

  values.resize(size);
  // Use the same initialization as the base layer
  const unsigned f_in = is_input ? num_inputs : num_neurons;
  const unsigned f_out = num_neurons;
  for (size_t i = 0; i < size; ++i)
  {
    values[i] = get_activation().weight_initialization(f_in, f_out);
  }
  grads.assign(size, 0.0);
  velocities.assign(size, 0.0);
  m1.assign(size, 0.0);
  m2.assign(size, 0.0);
  timesteps.assign(size, 0);
  decays.assign(size, weight_decay);
}

void GRURNNLayer::initialize_recurrent_weights(double weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const auto num_neurons = get_number_neurons();
  const auto num_inputs = get_number_input_neurons();

  const size_t num_rec_weights = static_cast<size_t>(num_neurons) * num_neurons;
  const size_t num_inp_weights = static_cast<size_t>(num_inputs) * num_neurons;

  // 1. Candidate State Recurrent Weights (using existing member)
  init_weights(_rw_values, _rw_grads, _rw_velocities, _rw_m1, _rw_m2, _rw_timesteps, _rw_decays, num_rec_weights, false, weight_decay);

  // 2. Update Gate (z)
  if (num_inputs > 0)
  {
    init_weights(_z_w_values, _z_w_grads, _z_w_velocities, _z_w_m1, _z_w_m2, _z_w_timesteps, _z_w_decays, num_inp_weights, true, weight_decay);
  }
  init_weights(_z_rw_values, _z_rw_grads, _z_rw_velocities, _z_rw_m1, _z_rw_m2, _z_rw_timesteps, _z_rw_decays, num_rec_weights, false, weight_decay);
  if (has_bias())
  {
    init_bias(_z_b_values, _z_b_grads, _z_b_velocities, _z_b_m1, _z_b_m2, _z_b_timesteps, _z_b_decays);
  }

  // 3. Reset Gate (r)
  if (num_inputs > 0)
  {
    init_weights(_r_w_values, _r_w_grads, _r_w_velocities, _r_w_m1, _r_w_m2, _r_w_timesteps, _r_w_decays, num_inp_weights, true, weight_decay);
  }
  init_weights(_r_rw_values, _r_rw_grads, _r_rw_velocities, _r_rw_m1, _r_rw_m2, _r_rw_timesteps, _r_rw_decays, num_rec_weights, false, weight_decay);
  if (has_bias())
  {
    init_bias(_r_b_values, _r_b_grads, _r_b_velocities, _r_b_m1, _r_b_m2, _r_b_timesteps, _r_b_decays);
  }
  cache_recurrent_weights();
}

void GRURNNLayer::calculate_forward_feed(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  if (batch_size == 0)
  {
    return;
  }

  const size_t N_prev = previous_layer.get_number_neurons();
  const size_t N_this = get_number_neurons();

  // 1. Flatten inputs [BatchSize x T x N_prev]
  thread_local std::vector<double> flattened_batch_inputs;
  size_t num_time_steps = 0;

  const unsigned prev_layer_index = previous_layer.get_layer_index();
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    if (!rnn_in.empty())
    {
      const size_t t = rnn_in.size() / N_prev;
      if (num_time_steps == 0) 
      {
        num_time_steps = t;
        flattened_batch_inputs.resize(batch_size * num_time_steps * N_prev);
      }
      std::copy(rnn_in.begin(), rnn_in.end(), flattened_batch_inputs.begin() + b * num_time_steps * N_prev);
    }
    else
    {
      const auto std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
      if (std_in.size() == N_prev)
      {
        if (num_time_steps == 0) 
        {
          num_time_steps = 1;
          flattened_batch_inputs.resize(batch_size * num_time_steps * N_prev);
        }
        for (size_t t = 0; t < num_time_steps; ++t)
        {
          std::copy(std_in.begin(), std_in.end(), flattened_batch_inputs.begin() + (b * num_time_steps + t) * N_prev);
        }
      }
    }
  }

  if (num_time_steps == 0)
  {
    return;
  }

  // 2. Pre-calculate Input-to-Gates (all 3 gates) for all ticks
  // Pre-activations buffer: [Batch x Ticks x 3 x N_this]
  thread_local std::vector<double> batch_pre_act;
  batch_pre_act.assign(batch_size * num_time_steps * GateCount * N_this, 0.0);

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  const unsigned int max_layer_threads = std::min(num_threads, 4U);
  const unsigned int active_threads = (num_threads > 1) ? std::max(1U, std::min(max_layer_threads, static_cast<unsigned int>((batch_size * num_time_steps * N_prev * N_this * 3) / 2000000))) : 1;
  const bool use_multithreading = (active_threads > 1);
  if (!use_multithreading)
  {
    pre_calculate_gates(0, batch_size, N_this, N_prev, num_time_steps, flattened_batch_inputs, batch_pre_act);
  }
  else
  {
    auto& flattened_batch_inputs_ref = flattened_batch_inputs;
    auto& batch_pre_act_ref = batch_pre_act;
    size_t start = 0;
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([start, end, N_this, N_prev, num_time_steps, &flattened_batch_inputs_ref, &batch_pre_act_ref, this]()
          {
            pre_calculate_gates(start, end, N_this, N_prev, num_time_steps, flattened_batch_inputs_ref, batch_pre_act_ref);
          });
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  // 3. Output sequence buffer and sequential recurrent pass
  std::vector<double> batch_output_sequences(batch_size * num_time_steps * N_this, 0.0);

  if (!use_multithreading)
  {
    run_forward_pass(
      0, 
      batch_size,
      N_this,
      num_time_steps,
      batch_pre_act,
      batch_residual_output_values,
      batch_output_sequences,
      batch_hidden_states,
      is_training
      );
  }
  else
  {
    auto& batch_pre_act_ref = batch_pre_act;
    size_t start = 0;
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([
          start, 
          end, 
          N_this,
          &batch_pre_act_ref,
          &batch_residual_output_values,
          &batch_output_sequences,
          &batch_hidden_states,
          is_training,
          num_time_steps,
          this]()
          {
            run_forward_pass(
              start, 
              end, 
              N_this,
              num_time_steps,
              batch_pre_act_ref,
              batch_residual_output_values,
              batch_output_sequences,
              batch_hidden_states,
              is_training
              );
          });
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  // 3. Store results
  for (size_t b = 0; b < batch_size; ++b)
  {
    const double* seq_ptr = &batch_output_sequences[b * num_time_steps * N_this];
    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), seq_ptr, num_time_steps * N_this);
    
    const double* last_ptr = &batch_output_sequences[(b * num_time_steps + num_time_steps - 1) * N_this];
    double* dest_ptr = batch_gradients_and_outputs[b].get_outputs_raw(get_layer_index());
    std::copy(last_ptr, last_ptr + N_this, dest_ptr);
  }
}

void GRURNNLayer::pre_calculate_gates(
  const size_t b_start, 
  const size_t b_end,
  const size_t N_this,
  const size_t N_prev,
  const size_t num_time_steps,
  const std::vector<double>& flattened_batch_inputs,
  std::vector<double>& batch_pre_act
) const
{
  const double* W_z = _z_w_values.data();
  const double* W_r = _r_w_values.data();
  const double* W_h = get_w_values().data();

  const size_t step_start = b_start * num_time_steps;
  const size_t step_end = b_end * num_time_steps;

  if (has_bias())
  {
    if (!_bias_cached.empty())
    {
      for (size_t step = step_start; step < step_end; ++step)
      {
        double* pre_t = &batch_pre_act[step * GateCount * N_this];
        std::copy(_bias_cached.begin(), _bias_cached.end(), pre_t);
      }
    }
    else
    {
      for (size_t step = step_start; step < step_end; ++step)
      {
        double* pre_t = &batch_pre_act[step * GateCount * N_this];
        std::copy(_z_b_values.begin(), _z_b_values.end(), pre_t);
        std::copy(_r_b_values.begin(), _r_b_values.end(), pre_t + N_this);
        std::copy(get_b_values().begin(), get_b_values().end(), pre_t + 2 * N_this);
      }
    }
  }
  else
  {
    std::fill(batch_pre_act.begin() + step_start * GateCount * N_this, batch_pre_act.begin() + step_end * GateCount * N_this, 0.0);
  }

  size_t step = step_start;
  for (; step + 3 < step_end; step += 4)
  {
    const double* x0 = &flattened_batch_inputs[step * N_prev];
    const double* x1 = &flattened_batch_inputs[(step + 1) * N_prev];
    const double* x2 = &flattened_batch_inputs[(step + 2) * N_prev];
    const double* x3 = &flattened_batch_inputs[(step + 3) * N_prev];

    double* y0_z = &batch_pre_act[step * GateCount * N_this];
    double* y0_r = y0_z + N_this;
    double* y0_h = y0_z + 2 * N_this;

    double* y1_z = &batch_pre_act[(step + 1) * GateCount * N_this];
    double* y1_r = y1_z + N_this;
    double* y1_h = y1_z + 2 * N_this;

    double* y2_z = &batch_pre_act[(step + 2) * GateCount * N_this];
    double* y2_r = y2_z + N_this;
    double* y2_h = y2_z + 2 * N_this;

    double* y3_z = &batch_pre_act[(step + 3) * GateCount * N_this];
    double* y3_r = y3_z + N_this;
    double* y3_h = y3_z + 2 * N_this;

    simd::gemm_four_batches(x0, x1, x2, x3, W_z, y0_z, y1_z, y2_z, y3_z, N_prev, N_this);
    simd::gemm_four_batches(x0, x1, x2, x3, W_r, y0_r, y1_r, y2_r, y3_r, N_prev, N_this);
    simd::gemm_four_batches(x0, x1, x2, x3, W_h, y0_h, y1_h, y2_h, y3_h, N_prev, N_this);
  }

  for (; step + 1 < step_end; step += 2)
  {
    const double* x0 = &flattened_batch_inputs[step * N_prev];
    const double* x1 = &flattened_batch_inputs[(step + 1) * N_prev];

    double* y0_z = &batch_pre_act[step * GateCount * N_this];
    double* y0_r = y0_z + N_this;
    double* y0_h = y0_z + 2 * N_this;

    double* y1_z = &batch_pre_act[(step + 1) * GateCount * N_this];
    double* y1_r = y1_z + N_this;
    double* y1_h = y1_z + 2 * N_this;

    simd::gemm_two_batches(x0, x1, W_z, y0_z, y1_z, N_prev, N_this);
    simd::gemm_two_batches(x0, x1, W_r, y0_r, y1_r, N_prev, N_this);
    simd::gemm_two_batches(x0, x1, W_h, y0_h, y1_h, N_prev, N_this);
  }

  for (; step < step_end; ++step)
  {
    const double* x_row = &flattened_batch_inputs[step * N_prev];
    double* y_z = &batch_pre_act[step * GateCount * N_this];
    double* y_r = y_z + N_this;
    double* y_h = y_z + 2 * N_this;

    simd::gemm_one_batch(x_row, W_z, y_z, N_prev, N_this);
    simd::gemm_one_batch(x_row, W_r, y_r, N_prev, N_this);
    simd::gemm_one_batch(x_row, W_h, y_h, N_prev, N_this);
  }
}

void GRURNNLayer::run_forward_pass(
  const size_t start,
  const size_t end,
  const size_t N_this,
  const size_t num_time_steps,
  const std::vector<double>& batch_pre_act,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<double>& batch_output_sequences,
  std::vector<HiddenStates>& batch_hidden_states,
  bool is_training
) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  std::vector<double> gated_h(N_this);
  std::vector<double> prev_h(N_this, 0.0);
  std::vector<double> current_h(N_this, 0.0);
  std::vector<double> packed_bptt_states(Multiplier * N_this); // Index [(Multiplier-1)*N_this, Multiplier*N_this) used for dropout mask

  for (size_t b = start; b < end; ++b)
  {
    // Reset hidden state for each sample in the batch!
    std::fill(prev_h.begin(), prev_h.end(), 0.0);
    std::fill(current_h.begin(), current_h.end(), 0.0);

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      // a. Retrieve precalculated Input-to-Gates (W * x_t + bias) and copy directly to packed_bptt_states
      const double* pre_t = &batch_pre_act[(b * num_time_steps + t) * GateCount * N_this];
      std::copy(pre_t, pre_t + 3 * N_this, packed_bptt_states.begin());

      double* z_ptr = packed_bptt_states.data();
      double* r_ptr = packed_bptt_states.data() + N_this;
      double* h_hat_pre_ptr = packed_bptt_states.data() + 2 * N_this;
      double* h_hat_activated_ptr = packed_bptt_states.data() + 3 * N_this;

      // c. Hidden-to-Gates (U * h_{t-1}) - Tiled
      const double* h_prev_ptr = prev_h.data();
      simd::gemv_add_two(_z_rw_values_T.data(), _r_rw_values_T.data(), h_prev_ptr, z_ptr, r_ptr, N_this, N_this);

      // d. Calculate Gates
      static const activation sigmoid_act(activation::method::sigmoid, 1.0);
      sigmoid_act.activate(z_ptr, z_ptr + 2 * N_this);

      // e. Candidate Recurrent State (U_h * (r * h_{t-1})) - Tiled
      simd::mul_vectors(r_ptr, h_prev_ptr, gated_h.data(), N_this);
      simd::gemv_add(_rw_values_T.data(), gated_h.data(), h_hat_pre_ptr, N_this, N_this);

      // f. Residuals and Candidate Activation
      if (!batch_residual_output_values.empty() && batch_residual_output_values[b].size() == N_this)
      {
        simd::add_vectors(batch_residual_output_values[b].data(), h_hat_pre_ptr, N_this);
      }

      std::copy(h_hat_pre_ptr, h_hat_pre_ptr + N_this, h_hat_activated_ptr);
      get_activation().activate(h_hat_activated_ptr, h_hat_activated_ptr + N_this, is_training);

      if (is_training && get_dropout() > 0.0)
      {
        const auto& neurons = get_neurons();
        double* mask_ptr = packed_bptt_states.data() + 4 * N_this;
        for (size_t j = 0; j < N_this; ++j)
        {
          double h_hat_activated_val = h_hat_activated_ptr[j];
          double mask = 1.0;
          double h_hat_final = h_hat_activated_val;
          const auto& neuron = neurons[j];
          if (neuron.is_dropout())
          {
            if (neuron.must_randomly_drop())
            {
              mask = 0.0;
              h_hat_final = 0.0;
            }
            else
            {
              mask = 1.0 / (1.0 - neuron.get_dropout_rate());
              h_hat_final *= mask;
            }
          }
          mask_ptr[j] = mask;
          current_h[j] = (1.0 - z_ptr[j]) * prev_h[j] + z_ptr[j] * h_hat_final;
          batch_output_sequences[(b * num_time_steps + t) * N_this + j] = current_h[j];
        }
      }
      else
      {
        std::fill_n(packed_bptt_states.begin() + 4 * N_this, N_this, 1.0);

        simd::gru_output_step(
          packed_bptt_states.data(),
          prev_h.data(),
          h_hat_activated_ptr,
          current_h.data(),
          &batch_output_sequences[(b * num_time_steps + t) * N_this],
          N_this
        );
      }

      batch_hidden_states[b].at(get_layer_index())[t].set_pre_activation_sums(packed_bptt_states.data(), packed_bptt_states.size());
      batch_hidden_states[b].at(get_layer_index())[t].set_hidden_state_values(current_h.data(), current_h.size());
      std::swap(prev_h, current_h);
    }
  }
}

void GRURNNLayer::calculate_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size) const
{
  (void)batch_gradients_and_outputs;
  (void)target_outputs_begin;
  (void)batch_hidden_states;
  (void)batch_size;
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  Logger::panic("GRURNNLayer: Trying to calculate output gradient with a non output layer!");
}

void GRURNNLayer::allocate_workspace()
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  if (_task_queue_pool == nullptr)
  {
    return;
  }
  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  allocate_workspace(num_threads);
}

void GRURNNLayer::allocate_workspace(unsigned int num_threads)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  if (_thread_workspaces.size() <= num_threads)
  {
    _thread_workspaces.resize(num_threads);
  }
  for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx)
  {
    if (!_thread_workspaces[thread_idx])
    {
      _thread_workspaces[thread_idx] = std::make_unique<BPTTWorkspace>();
    }
  }
}

GRURNNLayer::BPTTWorkspace& GRURNNLayer::get_workspace(size_t thread_idx) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
#if VALIDATE_DATA == 1
  if (thread_idx >= _thread_workspaces.size())
  {
    Logger::panic("Trying to get a workspace thread ", thread_idx, " past the workspaces size!");
  }
#endif
  return *_thread_workspaces[thread_idx];
}

void GRURNNLayer::calculate_bptt_batch_chunk(
    size_t start,
    size_t end,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& next_layer,
    const std::vector<std::vector<double>>& batch_next_grad_matrix,
    const std::vector<HiddenStates>& batch_hidden_states,
    int bptt_max_ticks,
    BPTTWorkspace& workspace,
    const BPTTWorkspace::AlignedVector& /*rw_values_T*/,
    const BPTTWorkspace::AlignedVector& /*z_rw_values_T*/,
    const BPTTWorkspace::AlignedVector& /*r_rw_values_T*/) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  const int t_start = static_cast<int>(num_time_steps) - 1;
  int t_end = (bptt_max_ticks > 0) ? std::max(0, t_start - bptt_max_ticks + 1) : 0;

  workspace.resize(N_this, N_prev, end - start, num_time_steps);

  // Step 1: Pre-calculate upstream gradients from next layer
  const size_t N_next = next_layer.get_number_neurons();
  const bool use_direct_gradients = batch_next_grad_matrix.empty();

  bool next_is_seq = false;
  if (use_direct_gradients)
  {
    if (end > start)
    {
      const auto next_grads = batch_gradients_and_outputs[start].get_gradients(next_layer.get_layer_index());
      next_is_seq = (next_grads.size() == num_time_steps * N_next);
    }
  }
  else
  {
    if (!batch_next_grad_matrix.empty())
    {
      next_is_seq = (batch_next_grad_matrix[0].size() == num_time_steps * N_next);
    }
  }

  double* grad_next_all_ptr = workspace.grad_from_next_all_t.data();
  const double* next_w_data = next_layer.get_w_values().data();

  for (size_t b_idx = 0; b_idx < end - start; ++b_idx)
  {
    size_t b = start + b_idx;
    const double* next_grad_matrix = nullptr;
    size_t next_grad_size = 0;
    if (use_direct_gradients)
    {
      const auto next_grads = batch_gradients_and_outputs[b].get_gradients(next_layer.get_layer_index());
      next_grad_matrix = next_grads.data();
      next_grad_size = next_grads.size();
    }
    else
    {
      if (b < batch_next_grad_matrix.size())
      {
        next_grad_matrix = batch_next_grad_matrix[b].data();
        next_grad_size = batch_next_grad_matrix[b].size();
      }
    }

    if (next_grad_matrix == nullptr)
    {
      continue;
    }

    for (int t = t_start; t >= t_end; --t)
    {
      const double* next_grad_ptr = nullptr;
      if (next_is_seq)
      {
        next_grad_ptr = &next_grad_matrix[t * N_next];
      }
      else if (t == t_start && next_grad_size == N_next)
      {
        next_grad_ptr = next_grad_matrix;
      }

      if (next_grad_ptr != nullptr)
      {
        double* dest_ptr = &grad_next_all_ptr[(b_idx * num_time_steps + t) * N_this];
        if (&next_layer == this)
        {
          // Identity mapping: next_grad_ptr is already dL/dh_{t+1}
          simd::add_vectors(next_grad_ptr, dest_ptr, N_this);
        }
        else
        {
          simd::gemv_add(next_w_data, next_grad_ptr, dest_ptr, N_this, N_next);
        }
      }
    }
  }

  // Step 2: BPTT Loop
  double* dz_ptr_all = workspace.chunk_dz.data();
  double* dr_ptr_all = workspace.chunk_dr.data();
  double* dh_hat_ptr_all = workspace.chunk_dh_hat.data();
  double* dh_prev_accum_ptr_all = workspace.chunk_dh_prev_accum.data();
  double* d_next_h_ptr_all = workspace.d_next_h.data();
  double* rnn_grad_ptr_all = workspace.rnn_grad_matrix.data();

  for (int t = t_start; t >= t_end; --t)
  {
    // Step 2a: Calculate gate gradients
    for (size_t b_idx = 0; b_idx < end - start; ++b_idx)
    {
      size_t b = start + b_idx;
      const auto& layer_states = batch_hidden_states[b].at(get_layer_index());
      const auto& state = layer_states[t];
      const auto& packed_states = state.get_pre_activation_sums();
      
      const double* z_vals = packed_states.data();
      const double* h_hat_pre_vals = &packed_states[2 * N_this];
      const double* h_hat_ptr = &packed_states[3 * N_this];
      const double* mask_vals = &packed_states[4 * N_this];
      const double* h_prev_vals = (t > 0) ? layer_states[t - 1].get_hidden_state_values().data() : nullptr;

      const double* grad_next_ptr = &grad_next_all_ptr[(b_idx * num_time_steps + t) * N_this];
      double* d_next_h_ptr = &d_next_h_ptr_all[b_idx * N_this];

      // Pre-calculate derivatives
      double* dh_hat_pre_deriv = &workspace.dh_hat_pre_deriv_buf[b_idx * N_this];
      const auto& act = get_activation();
      act.activate_derivative(h_hat_pre_vals, h_hat_pre_vals + N_this, h_hat_ptr, dh_hat_pre_deriv);

      simd::gru_bptt_gate_step(
        N_this,
        grad_next_ptr,
        d_next_h_ptr,
        z_vals,
        h_hat_ptr,
        h_prev_vals,
        h_hat_pre_vals,
        mask_vals,
        &dz_ptr_all[b_idx * N_this],
        &dh_hat_ptr_all[b_idx * N_this],
        &dh_prev_accum_ptr_all[b_idx * N_this],
        dh_hat_pre_deriv
      );
    }

    workspace.temp_Uh_T_dh_hat.assign((end - start) * N_this, 0.0);
    double* temp_Uh_ptr_all = workspace.temp_Uh_T_dh_hat.data();
    
    {
      size_t b = 0;
      const size_t batch_size_chunk = end - start;
      for (; b + 3 < batch_size_chunk; b += 4)
      {
        const double* dh0 = dh_hat_ptr_all + b * N_this;
        const double* dh1 = dh_hat_ptr_all + (b + 1) * N_this;
        const double* dh2 = dh_hat_ptr_all + (b + 2) * N_this;
        const double* dh3 = dh_hat_ptr_all + (b + 3) * N_this;

        double* y0 = temp_Uh_ptr_all + b * N_this;
        double* y1 = temp_Uh_ptr_all + (b + 1) * N_this;
        double* y2 = temp_Uh_ptr_all + (b + 2) * N_this;
        double* y3 = temp_Uh_ptr_all + (b + 3) * N_this;

        simd::gemm_four_batches(dh0, dh1, dh2, dh3, _rw_values_T.data(), y0, y1, y2, y3, N_this, N_this);
      }
      for (; b + 1 < batch_size_chunk; b += 2)
      {
        const double* dh0 = dh_hat_ptr_all + b * N_this;
        const double* dh1 = dh_hat_ptr_all + (b + 1) * N_this;

        double* y0 = temp_Uh_ptr_all + b * N_this;
        double* y1 = temp_Uh_ptr_all + (b + 1) * N_this;

        simd::gemm_two_batches(dh0, dh1, _rw_values_T.data(), y0, y1, N_this, N_this);
      }
      for (; b < batch_size_chunk; ++b)
      {
        const double* dh = dh_hat_ptr_all + b * N_this;
        double* y = temp_Uh_ptr_all + b * N_this;

        simd::gemm_one_batch(dh, _rw_values_T.data(), y, N_this, N_this);
      }
    }

    for (size_t b_idx = 0; b_idx < end - start; ++b_idx)
    {
      size_t b = start + b_idx;
      const auto& layer_states = batch_hidden_states[b].at(get_layer_index());
      const auto& packed_states = layer_states[t].get_pre_activation_sums();
      const double* r_vals = &packed_states[N_this];
      const double* h_prev_vals = (t > 0) ? layer_states[t - 1].get_hidden_state_values().data() : nullptr;
      const double* temp_Uh_ptr = &temp_Uh_ptr_all[b_idx * N_this];
      double* dr_ptr = &dr_ptr_all[b_idx * N_this];
      double* dz_ptr = &dz_ptr_all[b_idx * N_this];
      double* dh_hat_ptr = &dh_hat_ptr_all[b_idx * N_this];
      double* dh_prev_accum = &dh_prev_accum_ptr_all[b_idx * N_this];
      double* dh_next = &d_next_h_ptr_all[b_idx * N_this];

      // Calculate Reset gate gradients and backpropagate to previous hidden state
      simd::gru_bptt_reset_step(N_this, temp_Uh_ptr, h_prev_vals, r_vals, dh_prev_accum, dr_ptr, dh_next);

      double* dx_t = &workspace.dx_matrix[(b_idx * num_time_steps + t) * N_prev];
      std::fill_n(dx_t, N_prev, 0.0);

      double* dest = &rnn_grad_ptr_all[(b_idx * num_time_steps + t) * GateCount * N_this];
      std::copy(dh_hat_ptr, dh_hat_ptr + N_this, dest);
      std::copy(dz_ptr, dz_ptr + N_this, dest + N_this);
      std::copy(dr_ptr, dr_ptr + N_this, dest + 2 * N_this); // Gate 3 (Reset)
    }

    // Now run batched GEMM operations outside the batch loop:
    run_recurrent_gemm_backward(
      0, end - start, N_this,
      _z_rw_values_T.data(), _r_rw_values_T.data(),
      dz_ptr_all, dr_ptr_all,
      d_next_h_ptr_all
    );

    {
      size_t b = 0;
      const size_t batch_size_chunk = end - start;
      for (; b + 3 < batch_size_chunk; b += 4)
      {
        const double* z0 = dz_ptr_all + b * N_this;
        const double* z1 = dz_ptr_all + (b + 1) * N_this;
        const double* z2 = dz_ptr_all + (b + 2) * N_this;
        const double* z3 = dz_ptr_all + (b + 3) * N_this;

        const double* r0 = dr_ptr_all + b * N_this;
        const double* r1 = dr_ptr_all + (b + 1) * N_this;
        const double* r2 = dr_ptr_all + (b + 2) * N_this;
        const double* r3 = dr_ptr_all + (b + 3) * N_this;

        const double* h0 = dh_hat_ptr_all + b * N_this;
        const double* h1 = dh_hat_ptr_all + (b + 1) * N_this;
        const double* h2 = dh_hat_ptr_all + (b + 2) * N_this;
        const double* h3 = dh_hat_ptr_all + (b + 3) * N_this;

        double* y0 = &workspace.dx_matrix[(b * num_time_steps + t) * N_prev];
        double* y1 = &workspace.dx_matrix[((b + 1) * num_time_steps + t) * N_prev];
        double* y2 = &workspace.dx_matrix[((b + 2) * num_time_steps + t) * N_prev];
        double* y3 = &workspace.dx_matrix[((b + 3) * num_time_steps + t) * N_prev];

        simd::gemm_four_batches(z0, z1, z2, z3, _z_w_values_T.data(), y0, y1, y2, y3, N_this, N_prev);
        simd::gemm_four_batches(r0, r1, r2, r3, _r_w_values_T.data(), y0, y1, y2, y3, N_this, N_prev);
        simd::gemm_four_batches(h0, h1, h2, h3, _w_values_T.data(), y0, y1, y2, y3, N_this, N_prev);
      }
      for (; b + 1 < batch_size_chunk; b += 2)
      {
        const double* z0 = dz_ptr_all + b * N_this;
        const double* z1 = dz_ptr_all + (b + 1) * N_this;

        const double* r0 = dr_ptr_all + b * N_this;
        const double* r1 = dr_ptr_all + (b + 1) * N_this;

        const double* h0 = dh_hat_ptr_all + b * N_this;
        const double* h1 = dh_hat_ptr_all + (b + 1) * N_this;

        double* y0 = &workspace.dx_matrix[(b * num_time_steps + t) * N_prev];
        double* y1 = &workspace.dx_matrix[((b + 1) * num_time_steps + t) * N_prev];

        simd::gemm_two_batches(z0, z1, _z_w_values_T.data(), y0, y1, N_this, N_prev);
        simd::gemm_two_batches(r0, r1, _r_w_values_T.data(), y0, y1, N_this, N_prev);
        simd::gemm_two_batches(h0, h1, _w_values_T.data(), y0, y1, N_this, N_prev);
      }
      for (; b < batch_size_chunk; ++b)
      {
        const double* z = dz_ptr_all + b * N_this;
        const double* r = dr_ptr_all + b * N_this;
        const double* h = dh_hat_ptr_all + b * N_this;

        double* y = &workspace.dx_matrix[(b * num_time_steps + t) * N_prev];

        simd::gemm_one_batch(z, _z_w_values_T.data(), y, N_this, N_prev);
        simd::gemm_one_batch(r, _r_w_values_T.data(), y, N_this, N_prev);
        simd::gemm_one_batch(h, _w_values_T.data(), y, N_this, N_prev);
      }
    }
  }

  for (size_t b = start; b < end; ++b)
  {
    const size_t b_idx = b - start;
    const double* dX_src = &workspace.dx_matrix[b_idx * num_time_steps * N_prev];
    batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), dX_src, num_time_steps * N_prev);

    const double* gates_src = &workspace.rnn_grad_matrix[b_idx * num_time_steps * GateCount * N_this];
    batch_gradients_and_outputs[b].set_rnn_gate_gradients(get_layer_index(), gates_src, num_time_steps * GateCount * N_this);
  }
}

void GRURNNLayer::calculate_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<std::vector<double>>& batch_next_grad_matrix,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int bptt_max_ticks) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  if (batch_size == 0)
  {
    return;
  }

  const size_t N_this = get_number_neurons();
  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  if (num_time_steps == 0 || N_this == 0)
  {
    return;
  }

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  const size_t N_next = next_layer.get_number_neurons();
  const unsigned int max_layer_threads = std::min(num_threads, 4U);
  const unsigned int active_threads = (num_threads > 1) ? std::max(1U, std::min(max_layer_threads, static_cast<unsigned int>((batch_size * num_time_steps * N_this * (N_next + N_this) * 3) / 2000000))) : 1;
  const bool use_multithreading = (active_threads > 1);

  // Launch threads for each batch chunk
  if (!use_multithreading)
  {
    auto& workspace = get_workspace(0);
    calculate_bptt_batch_chunk(0, batch_size, batch_gradients_and_outputs, next_layer, batch_next_grad_matrix, batch_hidden_states, bptt_max_ticks, workspace, _rw_values_T, _z_rw_values_T, _r_rw_values_T);
  }
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([start, end, t, &batch_gradients_and_outputs, &next_layer, &batch_next_grad_matrix, &batch_hidden_states, bptt_max_ticks, this]()
          {
            auto& workspace = get_workspace(t);
            calculate_bptt_batch_chunk(start, end, batch_gradients_and_outputs, next_layer, batch_next_grad_matrix, batch_hidden_states, bptt_max_ticks, workspace, _rw_values_T, _z_rw_values_T, _r_rw_values_T);
          });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}

void GRURNNLayer::calculate_hidden_gradients_from_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<std::vector<double>>& batch_output_gradients,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int bptt_max_ticks) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const auto N_this = get_number_neurons();
  if (N_this == 0 || batch_size == 0)
  {
    return;
  }

  if (_identity_proxy == nullptr)
  {
    _identity_proxy = new FFLayer(0, N_this, N_this, 0.0, Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, false, 0.0);
    std::vector<double> id(static_cast<size_t>(N_this) * N_this, 0.0);
    for (unsigned i = 0; i < N_this; ++i)
    {
      id[i * N_this + i] = 1.0;
    }
    _identity_proxy->set_w_values(id);
  }

  calculate_hidden_gradients(batch_gradients_and_outputs, *_identity_proxy, batch_output_gradients, batch_hidden_states, batch_size, bptt_max_ticks);
}

double GRURNNLayer::get_recurrent_weight_value(unsigned from_neuron, unsigned to_neuron) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  return _rw_values[from_neuron * get_number_neurons() + to_neuron];
}

Layer* GRURNNLayer::clone() const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  return new GRURNNLayer(*this);
}

void GRURNNLayer::calculate_and_store_gradients(
    const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& hidden_states,
    const Layer& previous_layer,
    size_t batch_size,
    int bptt_max_ticks)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  if (batch_size == 0)
  {
    return;
  }

  const unsigned num_outputs = get_number_neurons();
  const unsigned num_inputs = get_number_input_neurons();
  const unsigned num_time_steps = (unsigned)hidden_states[0].at(get_layer_index()).size();
  const int t_start = static_cast<int>(num_time_steps) - 1;
  const int t_end = (bptt_max_ticks > 0) ? std::max(0, t_start - bptt_max_ticks + 1) : 0;

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  const size_t N_this = num_outputs;
  const size_t N_prev = previous_layer.get_number_neurons();
  const size_t T = num_time_steps;
  const unsigned int max_layer_threads = std::min(num_threads, 4U);
  const unsigned int active_threads = (num_threads > 1) ? std::max(1U, std::min(max_layer_threads, static_cast<unsigned int>((batch_size * T * N_this * (N_prev + N_this) * 3) / 2000000))) : 1;

  if (_thread_w_grads.size() < active_threads)
  {
    _thread_w_grads.resize(active_threads);
  }
  if (_thread_rw_grads.size() < active_threads)
  {
    _thread_rw_grads.resize(active_threads);
  }
  if (_thread_z_w_grads.size() < active_threads)
  {
    _thread_z_w_grads.resize(active_threads);
  }
  if (_thread_z_rw_grads.size() < active_threads)
  {
    _thread_z_rw_grads.resize(active_threads);
  }
  if (_thread_r_w_grads.size() < active_threads)
  {
    _thread_r_w_grads.resize(active_threads);
  }
  if (_thread_r_rw_grads.size() < active_threads)
  {
    _thread_r_rw_grads.resize(active_threads);
  }
  if (_thread_b_grads.size() < active_threads)
  {
    _thread_b_grads.resize(active_threads);
  }
  if (_thread_z_b_grads.size() < active_threads)
  {
    _thread_z_b_grads.resize(active_threads);
  }
  if (_thread_r_b_grads.size() < active_threads)
  {
    _thread_r_b_grads.resize(active_threads);
  }

  for (unsigned int t = 0; t < active_threads; ++t)
  {
    _thread_w_grads[t].resize(_w_grads.size());
    std::fill(_thread_w_grads[t].begin(), _thread_w_grads[t].end(), 0.0);
    _thread_rw_grads[t].resize(_rw_grads.size());
    std::fill(_thread_rw_grads[t].begin(), _thread_rw_grads[t].end(), 0.0);
    _thread_z_w_grads[t].resize(_z_w_grads.size());
    std::fill(_thread_z_w_grads[t].begin(), _thread_z_w_grads[t].end(), 0.0);
    _thread_z_rw_grads[t].resize(_z_rw_grads.size());
    std::fill(_thread_z_rw_grads[t].begin(), _thread_z_rw_grads[t].end(), 0.0);
    _thread_r_w_grads[t].resize(_r_w_grads.size());
    std::fill(_thread_r_w_grads[t].begin(), _thread_r_w_grads[t].end(), 0.0);
    _thread_r_rw_grads[t].resize(_r_rw_grads.size());
    std::fill(_thread_r_rw_grads[t].begin(), _thread_r_rw_grads[t].end(), 0.0);

    _thread_b_grads[t].resize(has_bias() ? num_outputs : 0);
    std::fill(_thread_b_grads[t].begin(), _thread_b_grads[t].end(), 0.0);
    _thread_z_b_grads[t].resize(has_bias() ? num_outputs : 0);
    std::fill(_thread_z_b_grads[t].begin(), _thread_z_b_grads[t].end(), 0.0);
    _thread_r_b_grads[t].resize(has_bias() ? num_outputs : 0);
    std::fill(_thread_r_b_grads[t].begin(), _thread_r_b_grads[t].end(), 0.0);
  }

  auto run_chunk = [&](size_t start, size_t end, size_t thread_idx)
  {
    auto& local_w_grads = _thread_w_grads[thread_idx];
    auto& local_rw_grads = _thread_rw_grads[thread_idx];
    auto& local_z_w_grads = _thread_z_w_grads[thread_idx];
    auto& local_z_rw_grads = _thread_z_rw_grads[thread_idx];
    auto& local_r_w_grads = _thread_r_w_grads[thread_idx];
    auto& local_r_rw_grads = _thread_r_rw_grads[thread_idx];
    auto& local_b_grads = _thread_b_grads[thread_idx];
    auto& local_z_b_grads = _thread_z_b_grads[thread_idx];
    auto& local_r_b_grads = _thread_r_b_grads[thread_idx];

    for (size_t b = start; b < end; ++b)
    {
      const auto& rnn_grads = batch_gradients_and_outputs[b].get_rnn_gate_gradients(get_layer_index());
      const auto& prev_outputs_rnn = batch_gradients_and_outputs[b].get_rnn_outputs(previous_layer.get_layer_index());
      const auto& prev_outputs_std = batch_gradients_and_outputs[b].get_outputs(previous_layer.get_layer_index());
      const auto& prev_outputs = !prev_outputs_rnn.empty() ? prev_outputs_rnn : prev_outputs_std;

      if (rnn_grads.size() != static_cast<size_t>(num_time_steps) * GateCount * num_outputs)
      {
        continue;
      }

      for (int t = t_start; t >= t_end; --t)
      {
        const size_t base_idx = t * GateCount * num_outputs;
        const double* gh = &rnn_grads[base_idx];
        const double* gz = &rnn_grads[base_idx + num_outputs];
        const double* gr = &rnn_grads[base_idx + 2 * num_outputs];

        const double* prev_input_ptr = nullptr;
        if (prev_outputs.size() == num_inputs)
        {
          prev_input_ptr = prev_outputs.data();
        }
        else if (prev_outputs.size() >= (t + 1) * num_inputs)
        {
          prev_input_ptr = &prev_outputs[t * num_inputs];
        }
        
        const double* prev_hidden_ptr = nullptr;
        if (t > 0)
        {
          prev_hidden_ptr = hidden_states[b].at(get_layer_index())[t - 1].get_hidden_state_values().data();
        }

        const auto& packed = hidden_states[b].at(get_layer_index())[t].get_pre_activation_sums();

        // Bias Gradients
        if (has_bias())
        {
          simd::add_vectors(gh, local_b_grads.data(), num_outputs);
          simd::add_vectors(gz, local_z_b_grads.data(), num_outputs);
          simd::add_vectors(gr, local_r_b_grads.data(), num_outputs);
        }

        // Weight Gradients (Outer Product) - Vectorized over num_outputs
        if (prev_input_ptr)
        {
          unsigned i = 0;
          for (; i + 3 < num_inputs; i += 4)
          {
            const double x0 = prev_input_ptr[i];
            const double x1 = prev_input_ptr[i + 1];
            const double x2 = prev_input_ptr[i + 2];
            const double x3 = prev_input_ptr[i + 3];

            simd::mul_add_four_scalars(x0, x1, x2, x3, gh, &local_w_grads[i * num_outputs], &local_w_grads[(i + 1) * num_outputs], &local_w_grads[(i + 2) * num_outputs], &local_w_grads[(i + 3) * num_outputs], num_outputs);
            simd::mul_add_four_scalars(x0, x1, x2, x3, gz, &local_z_w_grads[i * num_outputs], &local_z_w_grads[(i + 1) * num_outputs], &local_z_w_grads[(i + 2) * num_outputs], &local_z_w_grads[(i + 3) * num_outputs], num_outputs);
            simd::mul_add_four_scalars(x0, x1, x2, x3, gr, &local_r_w_grads[i * num_outputs], &local_r_w_grads[(i + 1) * num_outputs], &local_r_w_grads[(i + 2) * num_outputs], &local_r_w_grads[(i + 3) * num_outputs], num_outputs);
          }
          for (; i + 1 < num_inputs; i += 2)
          {
            const double x0 = prev_input_ptr[i];
            const double x1 = prev_input_ptr[i + 1];

            simd::mul_add_two_scalars(x0, x1, gh, &local_w_grads[i * num_outputs], &local_w_grads[(i + 1) * num_outputs], num_outputs);
            simd::mul_add_two_scalars(x0, x1, gz, &local_z_w_grads[i * num_outputs], &local_z_w_grads[(i + 1) * num_outputs], num_outputs);
            simd::mul_add_two_scalars(x0, x1, gr, &local_r_w_grads[i * num_outputs], &local_r_w_grads[(i + 1) * num_outputs], num_outputs);
          }
          for (; i < num_inputs; ++i)
          {
            simd::mul_add_three(prev_input_ptr[i], gh, gz, gr, &local_w_grads[i * num_outputs], &local_z_w_grads[i * num_outputs], &local_r_w_grads[i * num_outputs], num_outputs);
          }
        }

        // Recurrent Weight Gradients (Outer Product) - Vectorized over num_outputs
        if (prev_hidden_ptr)
        {
          const double* r_vals = &packed[num_outputs];
          unsigned k = 0;
          for (; k + 3 < num_outputs; k += 4)
          {
            const double hp0 = prev_hidden_ptr[k];
            const double hp1 = prev_hidden_ptr[k + 1];
            const double hp2 = prev_hidden_ptr[k + 2];
            const double hp3 = prev_hidden_ptr[k + 3];

            const double rv0 = r_vals[k];
            const double rv1 = r_vals[k + 1];
            const double rv2 = r_vals[k + 2];
            const double rv3 = r_vals[k + 3];

            simd::mul_add_four_scalars(rv0 * hp0, rv1 * hp1, rv2 * hp2, rv3 * hp3, gh, &local_rw_grads[k * num_outputs], &local_rw_grads[(k + 1) * num_outputs], &local_rw_grads[(k + 2) * num_outputs], &local_rw_grads[(k + 3) * num_outputs], num_outputs);
            simd::mul_add_four_scalars(hp0, hp1, hp2, hp3, gz, &local_z_rw_grads[k * num_outputs], &local_z_rw_grads[(k + 1) * num_outputs], &local_z_rw_grads[(k + 2) * num_outputs], &local_z_rw_grads[(k + 3) * num_outputs], num_outputs);
            simd::mul_add_four_scalars(hp0, hp1, hp2, hp3, gr, &local_r_rw_grads[k * num_outputs], &local_r_rw_grads[(k + 1) * num_outputs], &local_r_rw_grads[(k + 2) * num_outputs], &local_r_rw_grads[(k + 3) * num_outputs], num_outputs);
          }
          for (; k + 1 < num_outputs; k += 2)
          {
            const double hp0 = prev_hidden_ptr[k];
            const double hp1 = prev_hidden_ptr[k + 1];

            const double rv0 = r_vals[k];
            const double rv1 = r_vals[k + 1];

            simd::mul_add_two_scalars(rv0 * hp0, rv1 * hp1, gh, &local_rw_grads[k * num_outputs], &local_rw_grads[(k + 1) * num_outputs], num_outputs);
            simd::mul_add_two_scalars(hp0, hp1, gz, &local_z_rw_grads[k * num_outputs], &local_z_rw_grads[(k + 1) * num_outputs], num_outputs);
            simd::mul_add_two_scalars(hp0, hp1, gr, &local_r_rw_grads[k * num_outputs], &local_r_rw_grads[(k + 1) * num_outputs], num_outputs);
          }
          for (; k < num_outputs; ++k)
          {
            const double h_prev = prev_hidden_ptr[k];
            const double r_val = r_vals[k];
            simd::mul_add_three_scalars(
              r_val * h_prev, h_prev, h_prev,
              gh, gz, gr,
              &local_rw_grads[k * num_outputs],
              &local_z_rw_grads[k * num_outputs],
              &local_r_rw_grads[k * num_outputs],
              num_outputs
            );
          }
        }
      }
    }
  };

  const bool use_multithreading = (active_threads > 1);
  if (!use_multithreading)
  {
    run_chunk(0, batch_size, 0);
  }
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([start, end, t, &run_chunk]()
          { 
            run_chunk(start, end, t); 
          });
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  // Merge
  zero_gradients();
  for (unsigned int t = 0; t < active_threads; ++t)
  {
    simd::add_vectors(_thread_w_grads[t].data(), _w_grads.data(), _w_grads.size());
    simd::add_vectors(_thread_z_w_grads[t].data(), _z_w_grads.data(), _z_w_grads.size());
    simd::add_vectors(_thread_r_w_grads[t].data(), _r_w_grads.data(), _r_w_grads.size());

    simd::add_vectors(_thread_rw_grads[t].data(), _rw_grads.data(), _rw_grads.size());
    simd::add_vectors(_thread_z_rw_grads[t].data(), _z_rw_grads.data(), _z_rw_grads.size());
    simd::add_vectors(_thread_r_rw_grads[t].data(), _r_rw_grads.data(), _r_rw_grads.size());

    if (has_bias())
    {
      simd::add_vectors(_thread_b_grads[t].data(), _b_grads.data(), _b_grads.size());
      simd::add_vectors(_thread_z_b_grads[t].data(), _z_b_grads.data(), _z_b_grads.size());
      simd::add_vectors(_thread_r_b_grads[t].data(), _r_b_grads.data(), _r_b_grads.size());
    }
  }

  const double denom = static_cast<double>(batch_size);
  const double inv_batch = 1.0 / denom;
  
  const auto normalize = [inv_batch](std::vector<double>& grads)
  {
    simd::scale_vector(grads.data(), inv_batch, grads.size());
  };

  normalize(_w_grads);
  normalize(_z_w_grads);
  normalize(_r_w_grads);
  normalize(_rw_grads);
  normalize(_z_rw_grads);
  normalize(_r_rw_grads);
  if (has_bias())
  {
    normalize(_b_grads);
    normalize(_z_b_grads);
    normalize(_r_b_grads);
  }
}

double GRURNNLayer::get_gradient_norm_sq() const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  double norm_sq = 0.0;
  norm_sq += simd::sum_sq(_w_grads.data(), _w_grads.size());
  norm_sq += simd::sum_sq(_rw_grads.data(), _rw_grads.size());
  norm_sq += simd::sum_sq(_z_w_grads.data(), _z_w_grads.size());
  norm_sq += simd::sum_sq(_z_rw_grads.data(), _z_rw_grads.size());
  norm_sq += simd::sum_sq(_r_w_grads.data(), _r_w_grads.size());
  norm_sq += simd::sum_sq(_r_rw_grads.data(), _r_rw_grads.size());
  if (has_bias())
  {
    norm_sq += simd::sum_sq(_b_grads.data(), _b_grads.size());
    norm_sq += simd::sum_sq(_z_b_grads.data(), _z_b_grads.size());
    norm_sq += simd::sum_sq(_r_b_grads.data(), _r_b_grads.size());
  }

  return norm_sq;
}

void GRURNNLayer::zero_gradients()
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  Layer::zero_gradients();
  std::fill(_rw_grads.begin(), _rw_grads.end(), 0.0);
  std::fill(_z_w_grads.begin(), _z_w_grads.end(), 0.0);
  std::fill(_z_rw_grads.begin(), _z_rw_grads.end(), 0.0);
  std::fill(_z_b_grads.begin(), _z_b_grads.end(), 0.0);
  std::fill(_r_w_grads.begin(), _r_w_grads.end(), 0.0);
  std::fill(_r_rw_grads.begin(), _r_rw_grads.end(), 0.0);
  std::fill(_r_b_grads.begin(), _r_b_grads.end(), 0.0);
}

void GRURNNLayer::apply_stored_gradients(double learning_rate, double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  
  auto app = [&](std::vector<double>& v, std::vector<double>& g, std::vector<double>& vel, std::vector<double>& m1, std::vector<double>& m2, std::vector<long long>& ts, const std::vector<double>& dec, bool is_bias) {
    apply_update_to_vector(v, g, vel, m1, m2, ts, dec, learning_rate, clipping_scale, is_bias, get_optimiser_type());
  };

  // 1. Candidate State
  app(_w_values, _w_grads, _w_velocities, _w_m1, _w_m2, _w_timesteps, _w_decays, false);
  app(_rw_values, _rw_grads, _rw_velocities, _rw_m1, _rw_m2, _rw_timesteps, _rw_decays, false);
  if (has_bias()) app(_b_values, _b_grads, _b_velocities, _b_m1, _b_m2, _b_timesteps, _b_decays, true);

  // 2. Update Gate (z)
  app(_z_w_values, _z_w_grads, _z_w_velocities, _z_w_m1, _z_w_m2, _z_w_timesteps, _z_w_decays, false);
  app(_z_rw_values, _z_rw_grads, _z_rw_velocities, _z_rw_m1, _z_rw_m2, _z_rw_timesteps, _z_rw_decays, false);
  if (has_bias()) app(_z_b_values, _z_b_grads, _z_b_velocities, _z_b_m1, _z_b_m2, _z_b_timesteps, _z_b_decays, true);

  // 3. Reset Gate (r)
  app(_r_w_values, _r_w_grads, _r_w_velocities, _r_w_m1, _r_w_m2, _r_w_timesteps, _r_w_decays, false);
  app(_r_rw_values, _r_rw_grads, _r_rw_velocities, _r_rw_m1, _r_rw_m2, _r_rw_timesteps, _r_rw_decays, false);
  if (has_bias()) app(_r_b_values, _r_b_grads, _r_b_velocities, _r_b_m1, _r_b_m2, _r_b_timesteps, _r_b_decays, true);

  cache_recurrent_weights();
  zero_gradients();
}

void GRURNNLayer::set_number_of_threads(int number_of_threads)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  Layer::set_number_of_threads(number_of_threads);
  allocate_workspace();
}

void GRURNNLayer::cache_recurrent_weights()
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t n = get_number_neurons();
  const size_t n_prev = get_number_input_neurons();
  if (n == 0)
  {
    return;
  }

  _rw_values_T.resize(n * n);
  _z_rw_values_T.resize(n * n);
  _r_rw_values_T.resize(n * n);

  for (size_t i = 0; i < n; ++i)
  {
    for (size_t j = 0; j < n; ++j)
    {
      _rw_values_T[j * n + i] = _rw_values[i * n + j];
      _z_rw_values_T[j * n + i] = _z_rw_values[i * n + j];
      _r_rw_values_T[j * n + i] = _r_rw_values[i * n + j];
    }
  }

  if (n_prev > 0)
  {
    _w_values_T.resize(n * n_prev);
    _z_w_values_T.resize(n * n_prev);
    _r_w_values_T.resize(n * n_prev);
    for (size_t i = 0; i < n_prev; ++i)
    {
      for (size_t j = 0; j < n; ++j)
      {
        _w_values_T[j * n_prev + i] = get_w_values()[i * n + j];
        _z_w_values_T[j * n_prev + i] = _z_w_values[i * n + j];
        _r_w_values_T[j * n_prev + i] = _r_w_values[i * n + j];
      }
    }
  }

  if (has_bias() && !_z_b_values.empty() && !_r_b_values.empty() && !get_b_values().empty())
  {
    _bias_cached.resize(3 * n);
    std::copy(_z_b_values.begin(), _z_b_values.end(), _bias_cached.begin());
    std::copy(_r_b_values.begin(), _r_b_values.end(), _bias_cached.begin() + n);
    std::copy(get_b_values().begin(), get_b_values().end(), _bias_cached.begin() + 2 * n);
  }
}

void GRURNNLayer::set_w_values(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t expected_size = N_this * N_prev;
  if (v.size() == expected_size * GateCount)
  {
    _z_w_values.assign(v.begin(), v.begin() + expected_size);
    _r_w_values.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    Layer::set_w_values(std::vector<double>(v.begin() + 2 * expected_size, v.end()));
  }
  else
  {
    Layer::set_w_values(v);
  }
  cache_recurrent_weights();
}


void GRURNNLayer::set_w_grads(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t expected_size = N_this * N_prev;
  if (v.size() == expected_size * GateCount)
  {
    _z_w_grads.assign(v.begin(), v.begin() + expected_size);
    _r_w_grads.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    Layer::set_w_grads(std::vector<double>(v.begin() + 2 * expected_size, v.end()));
  }
  else
  {
    Layer::set_w_grads(v);
  }
}

void GRURNNLayer::set_w_velocities(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t expected_size = N_this * N_prev;
  if (v.size() == expected_size * GateCount)
  {
    _z_w_velocities.assign(v.begin(), v.begin() + expected_size);
    _r_w_velocities.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    Layer::set_w_velocities(std::vector<double>(v.begin() + 2 * expected_size, v.end()));
  }
  else
  {
    Layer::set_w_velocities(v);
  }
}

void GRURNNLayer::set_w_m1(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t expected_size = N_this * N_prev;
  if (v.size() == expected_size * GateCount)
  {
    _z_w_m1.assign(v.begin(), v.begin() + expected_size);
    _r_w_m1.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    Layer::set_w_m1(std::vector<double>(v.begin() + 2 * expected_size, v.end()));
  }
  else
  {
    Layer::set_w_m1(v);
  }
}

void GRURNNLayer::set_w_m2(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t expected_size = N_this * N_prev;
  if (v.size() == expected_size * GateCount)
  {
    _z_w_m2.assign(v.begin(), v.begin() + expected_size);
    _r_w_m2.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    Layer::set_w_m2(std::vector<double>(v.begin() + 2 * expected_size, v.end()));
  }
  else
  {
    Layer::set_w_m2(v);
  }
}

void GRURNNLayer::set_w_timesteps(const std::vector<long long>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t expected_size = N_this * N_prev;
  if (v.size() == expected_size * GateCount)
  {
    _z_w_timesteps.assign(v.begin(), v.begin() + expected_size);
    _r_w_timesteps.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    Layer::set_w_timesteps(std::vector<long long>(v.begin() + 2 * expected_size, v.end()));
  }
  else
  {
    Layer::set_w_timesteps(v);
  }
}

void GRURNNLayer::set_w_decays(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t expected_size = N_this * N_prev;
  if (v.size() == expected_size * GateCount)
  {
    _z_w_decays.assign(v.begin(), v.begin() + expected_size);
    _r_w_decays.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    Layer::set_w_decays(std::vector<double>(v.begin() + 2 * expected_size, v.end()));
  }
  else
  {
    Layer::set_w_decays(v);
  }
}

void GRURNNLayer::set_b_values(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  if (v.size() == N_this * GateCount)
  {
    _z_b_values.assign(v.begin(), v.begin() + N_this);
    _r_b_values.assign(v.begin() + N_this, v.begin() + 2 * N_this);
    Layer::set_b_values(std::vector<double>(v.begin() + 2 * N_this, v.end()));
  }
  else
  {
    Layer::set_b_values(v);
  }
  cache_recurrent_weights();
}

void GRURNNLayer::set_b_grads(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  if (v.size() == N_this * GateCount)
  {
    _z_b_grads.assign(v.begin(), v.begin() + N_this);
    _r_b_grads.assign(v.begin() + N_this, v.begin() + 2 * N_this);
    Layer::set_b_grads(std::vector<double>(v.begin() + 2 * N_this, v.end()));
  }
  else
  {
    Layer::set_b_grads(v);
  }
}

void GRURNNLayer::set_b_velocities(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  if (v.size() == N_this * GateCount)
  {
    _z_b_velocities.assign(v.begin(), v.begin() + N_this);
    _r_b_velocities.assign(v.begin() + N_this, v.begin() + 2 * N_this);
    Layer::set_b_velocities(std::vector<double>(v.begin() + 2 * N_this, v.end()));
  }
  else
  {
    Layer::set_b_velocities(v);
  }
}

void GRURNNLayer::set_b_m1(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  if (v.size() == N_this * GateCount)
  {
    _z_b_m1.assign(v.begin(), v.begin() + N_this);
    _r_b_m1.assign(v.begin() + N_this, v.begin() + 2 * N_this);
    Layer::set_b_m1(std::vector<double>(v.begin() + 2 * N_this, v.end()));
  }
  else
  {
    Layer::set_b_m1(v);
  }
}

void GRURNNLayer::set_b_m2(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  if (v.size() == N_this * GateCount)
  {
    _z_b_m2.assign(v.begin(), v.begin() + N_this);
    _r_b_m2.assign(v.begin() + N_this, v.begin() + 2 * N_this);
    Layer::set_b_m2(std::vector<double>(v.begin() + 2 * N_this, v.end()));
  }
  else
  {
    Layer::set_b_m2(v);
  }
}

void GRURNNLayer::set_b_timesteps(const std::vector<long long>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  if (v.size() == N_this * GateCount)
  {
    _z_b_timesteps.assign(v.begin(), v.begin() + N_this);
    _r_b_timesteps.assign(v.begin() + N_this, v.begin() + 2 * N_this);
    Layer::set_b_timesteps(std::vector<long long>(v.begin() + 2 * N_this, v.end()));
  }
  else
  {
    Layer::set_b_timesteps(v);
  }
}

void GRURNNLayer::set_b_decays(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  if (v.size() == N_this * GateCount)
  {
    _z_b_decays.assign(v.begin(), v.begin() + N_this);
    _r_b_decays.assign(v.begin() + N_this, v.begin() + 2 * N_this);
    Layer::set_b_decays(std::vector<double>(v.begin() + 2 * N_this, v.end()));
  }
  else
  {
    Layer::set_b_decays(v);
  }
}

void GRURNNLayer::set_rw_values(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t expected_size = N_this * N_this;
  if (v.size() == expected_size * GateCount)
  {
    _z_rw_values.assign(v.begin(), v.begin() + expected_size);
    _r_rw_values.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _rw_values.assign(v.begin() + 2 * expected_size, v.end());
  }
  else
  {
    _rw_values = v;
  }
  cache_recurrent_weights();
}

void GRURNNLayer::set_rw_grads(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t expected_size = N_this * N_this;
  if (v.size() == expected_size * GateCount)
  {
    _z_rw_grads.assign(v.begin(), v.begin() + expected_size);
    _r_rw_grads.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _rw_grads.assign(v.begin() + 2 * expected_size, v.end());
  }
  else
  {
    _rw_grads = v;
  }
}

void GRURNNLayer::set_rw_velocities(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t expected_size = N_this * N_this;
  if (v.size() == expected_size * GateCount)
  {
    _z_rw_velocities.assign(v.begin(), v.begin() + expected_size);
    _r_rw_velocities.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _rw_velocities.assign(v.begin() + 2 * expected_size, v.end());
  }
  else
  {
    _rw_velocities = v;
  }
}

void GRURNNLayer::set_rw_m1(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t expected_size = N_this * N_this;
  if (v.size() == expected_size * GateCount)
  {
    _z_rw_m1.assign(v.begin(), v.begin() + expected_size);
    _r_rw_m1.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _rw_m1.assign(v.begin() + 2 * expected_size, v.end());
  }
  else
  {
    _rw_m1 = v;
  }
}

void GRURNNLayer::set_rw_m2(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t expected_size = N_this * N_this;
  if (v.size() == expected_size * GateCount)
  {
    _z_rw_m2.assign(v.begin(), v.begin() + expected_size);
    _r_rw_m2.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _rw_m2.assign(v.begin() + 2 * expected_size, v.end());
  }
  else
  {
    _rw_m2 = v;
  }
}

void GRURNNLayer::set_rw_timesteps(const std::vector<long long>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t expected_size = N_this * N_this;
  if (v.size() == expected_size * GateCount)
  {
    _z_rw_timesteps.assign(v.begin(), v.begin() + expected_size);
    _r_rw_timesteps.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _rw_timesteps.assign(v.begin() + 2 * expected_size, v.end());
  }
  else
  {
    _rw_timesteps = v;
  }
}

void GRURNNLayer::set_rw_decays(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t expected_size = N_this * N_this;
  if (v.size() == expected_size * GateCount)
  {
    _z_rw_decays.assign(v.begin(), v.begin() + expected_size);
    _r_rw_decays.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _rw_decays.assign(v.begin() + 2 * expected_size, v.end());
  }
  else
  {
    _rw_decays = v;
  }
}

void GRURNNLayer::run_recurrent_gemm_backward(
  size_t b_start,
  size_t b_end,
  size_t N_this,
  const double* U_z_T,
  const double* U_r_T,
  const double* dz_batch,
  const double* dr_batch,
  double* dh_next_batch) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  size_t b = b_start;
  for (; b + 3 < b_end; b += 4)
  {
    const double* z0 = dz_batch + b * N_this;
    const double* z1 = dz_batch + (b + 1) * N_this;
    const double* z2 = dz_batch + (b + 2) * N_this;
    const double* z3 = dz_batch + (b + 3) * N_this;

    const double* r0 = dr_batch + b * N_this;
    const double* r1 = dr_batch + (b + 1) * N_this;
    const double* r2 = dr_batch + (b + 2) * N_this;
    const double* r3 = dr_batch + (b + 3) * N_this;

    double* y0 = dh_next_batch + b * N_this;
    double* y1 = dh_next_batch + (b + 1) * N_this;
    double* y2 = dh_next_batch + (b + 2) * N_this;
    double* y3 = dh_next_batch + (b + 3) * N_this;

    simd::gemm_four_batches(z0, z1, z2, z3, U_z_T, y0, y1, y2, y3, N_this, N_this);
    simd::gemm_four_batches(r0, r1, r2, r3, U_r_T, y0, y1, y2, y3, N_this, N_this);
  }

  for (; b + 1 < b_end; b += 2)
  {
    const double* z0 = dz_batch + b * N_this;
    const double* z1 = dz_batch + (b + 1) * N_this;

    const double* r0 = dr_batch + b * N_this;
    const double* r1 = dr_batch + (b + 1) * N_this;

    double* y0 = dh_next_batch + b * N_this;
    double* y1 = dh_next_batch + (b + 1) * N_this;

    simd::gemm_two_batches(z0, z1, U_z_T, y0, y1, N_this, N_this);
    simd::gemm_two_batches(r0, r1, U_r_T, y0, y1, N_this, N_this);
  }

  for (; b < b_end; ++b)
  {
    const double* z = dz_batch + b * N_this;
    const double* r = dr_batch + b * N_this;
    double* y = dh_next_batch + b * N_this;

    simd::gemm_one_batch(z, U_z_T, y, N_this, N_this);
    simd::gemm_one_batch(r, U_r_T, y, N_this, N_this);
  }
}

} // namespace myoddweb::nn

