#include "./libraries/instrumentor.h"
#include "grurnnlayer.h"
#include "fflayer.h"
#include "logger.h"
#include "simd_utils.h"
#include <immintrin.h>
#include <numeric>

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
  _thread_workspaces(std::move(src._thread_workspaces))
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
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
    allocate_workspace();
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
    _thread_workspaces = std::move(src._thread_workspaces);
  }
  return *this;
}

GRURNNLayer::~GRURNNLayer()
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
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
  std::vector<double> flattened_batch_inputs;
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

  // 2. Output sequence buffer
  std::vector<double> batch_output_sequences(batch_size * num_time_steps * N_this, 0.0);

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads <= 1)
  {
    run_forward_pass(
      0, 
      batch_size,
      N_this,
      N_prev,
      num_time_steps,
      flattened_batch_inputs,
      batch_residual_output_values,
      batch_output_sequences,
      batch_hidden_states,
      is_training
      );
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
        _task_queue_pool->enqueue([
          start, 
          end, 
          N_this,
          &flattened_batch_inputs,
          &batch_residual_output_values,
          &batch_output_sequences,
          &batch_hidden_states,
          is_training,
          num_time_steps,
          N_prev,
          this]()
          {
            run_forward_pass(
              start, 
              end, 
              N_this,
              N_prev,
              num_time_steps,
              flattened_batch_inputs,
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
    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), std::vector<double>(seq_ptr, seq_ptr + num_time_steps * N_this));
    
    const double* last_ptr = &batch_output_sequences[(b * num_time_steps + num_time_steps - 1) * N_this];
    batch_gradients_and_outputs[b].set_outputs(get_layer_index(), std::vector<double>(last_ptr, last_ptr + N_this));
  }
}

void GRURNNLayer::run_forward_pass(
  const size_t start,
  const size_t end,
  const size_t N_this,
  const size_t N_prev,
  const size_t num_time_steps,
  const std::vector<double>& flattened_batch_inputs,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<double>& batch_output_sequences,
  std::vector<HiddenStates>& batch_hidden_states,
  bool is_training
) const
{
  std::vector<double> z_pre(N_this), r_pre(N_this), h_hat_pre(N_this);
  std::vector<double> packed_bptt_states(Multiplier * N_this); // Index [(Multiplier-1)*N_this, Multiplier*N_this) used for dropout mask

  const double* W_z = _z_w_values.data();
  const double* W_r = _r_w_values.data();
  const double* W_h = get_w_values().data();
  const double* U_z = _z_rw_values.data();
  const double* U_r = _r_rw_values.data();
  const double* U_h = _rw_values.data();

  for (size_t b = start; b < end; ++b)
  {
    // Reset hidden state for each sample in the batch!
    std::vector<double> prev_h(N_this, 0.0);
    std::vector<double> current_h(N_this, 0.0);

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      // a. Initialize with bias
      if (has_bias())
      {
        std::copy(_z_b_values.begin(), _z_b_values.end(), z_pre.begin());
        std::copy(_r_b_values.begin(), _r_b_values.end(), r_pre.begin());
        std::copy(_b_values.begin(), _b_values.end(), h_hat_pre.begin());
      }
      else
      {
        std::fill(z_pre.begin(), z_pre.end(), 0.0);
        std::fill(r_pre.begin(), r_pre.end(), 0.0);
        std::fill(h_hat_pre.begin(), h_hat_pre.end(), 0.0);
      }

      // b. Input-to-Gates (W * x_t) - Tiled
      const double* x_t = &flattened_batch_inputs[(b * num_time_steps + t) * N_prev];
      constexpr size_t BLOCK_SIZE = 64;
      for (size_t i0 = 0; i0 < N_prev; i0 += BLOCK_SIZE)
      {
        size_t i_limit = std::min(i0 + BLOCK_SIZE, N_prev);
        for (size_t j0 = 0; j0 < N_this; j0 += BLOCK_SIZE)
        {
          size_t j_limit = std::min(j0 + BLOCK_SIZE, N_this);
          for (size_t i = i0; i < i_limit; ++i)
          {
            const double x_val = x_t[i];
            if (x_val == 0.0) continue;
            for (size_t j = j0; j < j_limit; ++j)
            {
              z_pre[j] += x_val * W_z[i * N_this + j];
              r_pre[j] += x_val * W_r[i * N_this + j];
              h_hat_pre[j] += x_val * W_h[i * N_this + j];
            }
          }
        }
      }

      // c. Hidden-to-Gates (U * h_{t-1}) - Tiled
      const double* h_prev_ptr = prev_h.data();
      for (size_t i0 = 0; i0 < N_this; i0 += BLOCK_SIZE)
      {
        size_t i_limit = std::min(i0 + BLOCK_SIZE, N_this);
        for (size_t j0 = 0; j0 < N_this; j0 += BLOCK_SIZE)
        {
          size_t j_limit = std::min(j0 + BLOCK_SIZE, N_this);
          for (size_t i = i0; i < i_limit; ++i)
          {
            const double h_val = h_prev_ptr[i];
            if (h_val == 0.0) continue;
            for (size_t j = j0; j < j_limit; ++j)
            {
              z_pre[j] += h_val * U_z[i * N_this + j];
              r_pre[j] += h_val * U_r[i * N_this + j];
            }
          }
        }
      }

      // d. Calculate Gates
      for (size_t j = 0; j < N_this; ++j)
      {
        double z = 1.0 / (1.0 + std::exp(-z_pre[j]));
        double r = 1.0 / (1.0 + std::exp(-r_pre[j]));
        packed_bptt_states[j] = z;
        packed_bptt_states[N_this + j] = r;
      }

      // e. Candidate Recurrent State (U_h * (r * h_{t-1})) - Tiled
      for (size_t i0 = 0; i0 < N_this; i0 += BLOCK_SIZE)
      {
        size_t i_limit = std::min(i0 + BLOCK_SIZE, N_this);
        for (size_t j0 = 0; j0 < N_this; j0 += BLOCK_SIZE)
        {
          size_t j_limit = std::min(j0 + BLOCK_SIZE, N_this);
          for (size_t i = i0; i < i_limit; ++i)
          {
            const double gated_h = packed_bptt_states[N_this + i] * h_prev_ptr[i];
            if (gated_h == 0.0)
            {
              continue;
            }
            for (size_t j = j0; j < j_limit; ++j)
            {
              h_hat_pre[j] += gated_h * U_h[i * N_this + j];
            }
          }
        }
      }

      // f. Residuals and Candidate Activation
      if (!batch_residual_output_values.empty() && batch_residual_output_values[b].size() == N_this)
      {
        for (size_t j = 0; j < N_this; ++j)
        {
          h_hat_pre[j] += batch_residual_output_values[b][j];
        }
      }

      std::vector<double> h_hat_vec = h_hat_pre;
      get_activation().activate(h_hat_vec.data(), h_hat_vec.data() + N_this, is_training);

      // g. Final State Update and Dropout
      for (size_t j = 0; j < N_this; ++j)
      {
        packed_bptt_states[2 * N_this + j] = h_hat_pre[j];

        double mask = 1.0;
        double h_hat = h_hat_vec[j];
        if (is_training && get_neuron((unsigned)j).is_dropout())
        {
          const auto& neuron = get_neuron((unsigned)j);
          if (neuron.must_randomly_drop())
          {
            mask = 0.0;
            h_hat = 0.0;
          }
          else
          {
            mask = 1.0 / (1.0 - neuron.get_dropout_rate());
            h_hat *= mask;
          }
        }
        packed_bptt_states[3 * N_this + j] = h_hat;
        packed_bptt_states[4 * N_this + j] = mask;
        current_h[j] = (1.0 - packed_bptt_states[j]) * prev_h[j] + packed_bptt_states[j] * h_hat;
        batch_output_sequences[(b * num_time_steps + t) * N_this + j] = current_h[j];
      }

      batch_hidden_states[b].at(get_layer_index())[t].set_pre_activation_sums(packed_bptt_states);
      batch_hidden_states[b].at(get_layer_index())[t].set_hidden_state_values(current_h);
      prev_h = current_h;
    }
  }
}

void GRURNNLayer::calculate_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size) const
{
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
  const bool next_is_seq = (batch_next_grad_matrix[0].size() == num_time_steps * N_next);

  double* grad_next_all_ptr = workspace.grad_from_next_all_t.data();

  for (size_t b_idx = 0; b_idx < end - start; ++b_idx)
  {
    size_t b = start + b_idx;
    const double* next_grad_matrix = batch_next_grad_matrix[b].data();

    for (int t = t_start; t >= t_end; --t)
    {
      const double* next_grad_ptr = nullptr;
      if (next_is_seq)
      {
        next_grad_ptr = &next_grad_matrix[t * N_next];
      }
      else if (t == t_start && batch_next_grad_matrix[b].size() == N_next)
      {
        next_grad_ptr = next_grad_matrix;
      }

      if (next_grad_ptr)
      {
        double* dest_ptr = &grad_next_all_ptr[(b_idx * num_time_steps + t) * N_this];
        if (&next_layer == this)
        {
          // Identity mapping: next_grad_ptr is already dL/dh_{t+1}
          for (size_t i = 0; i < N_this; ++i)
          {
            dest_ptr[i] += next_grad_ptr[i];
          }
        }
        else
        {
          for (size_t i = 0; i < N_this; ++i)
          {
            const double* next_w_row = next_layer.get_w_values().data() + i * N_next;
            dest_ptr[i] += simd::dot_product(next_grad_ptr, next_w_row, N_next);
          }
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
      const auto& h_vals = state.get_hidden_state_values();
      const auto& packed_states = state.get_pre_activation_sums();
      
      const double* z_vals = packed_states.data();
      const double* r_vals = &packed_states[N_this];
      const double* h_hat_pre_vals = &packed_states[2 * N_this];
      const double* h_hat_ptr = &packed_states[3 * N_this];
      const double* mask_vals = &packed_states[4 * N_this];
      const double* h_prev_vals = (t > 0) ? layer_states[t - 1].get_hidden_state_values().data() : nullptr;

      const double* grad_next_ptr = &grad_next_all_ptr[(b_idx * num_time_steps + t) * N_this];
      double* d_next_h_ptr = &d_next_h_ptr_all[b_idx * N_this];

      simd::gru_bptt_gate_step(
        N_this,
        grad_next_ptr,
        d_next_h_ptr,
        z_vals,
        h_hat_ptr,
        h_prev_vals,
        h_hat_pre_vals,
        &dz_ptr_all[b_idx * N_this],
        &dh_hat_ptr_all[b_idx * N_this],
        &dh_prev_accum_ptr_all[b_idx * N_this],
        [&](double x) { return get_activation().activate_derivative(x); }
      );

      // Apply dropout mask to candidate state pre-activation gradient
      double* dh_hat_pre = &dh_hat_ptr_all[b_idx * N_this];
      for (size_t j = 0; j < N_this; ++j)
      {
        dh_hat_pre[j] *= mask_vals[j];
      }
    }

    workspace.temp_Uh_T_dh_hat.assign((end - start) * N_this, 0.0);
    double* temp_Uh_ptr_all = workspace.temp_Uh_T_dh_hat.data();
    for (size_t b_idx = 0; b_idx < end - start; ++b_idx)
    {
      const double* dh_hat_ptr = &dh_hat_ptr_all[b_idx * N_this];
      for (size_t k = 0; k < N_this; ++k)
      {
        temp_Uh_ptr_all[b_idx * N_this + k] = simd::dot_product(dh_hat_ptr, &_rw_values[k * N_this], N_this);
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

      for (size_t k = 0; k < N_this; ++k)
      {
        double grad_rh = temp_Uh_ptr[k];
        double h_prev = (h_prev_vals) ? h_prev_vals[k] : 0.0;
        double r = r_vals[k];
        double d_r_pre = grad_rh * h_prev * r * (1.0 - r);
        dr_ptr[k] = d_r_pre;
        dh_next[k] = dh_prev_accum[k] + grad_rh * r + 
                          simd::dot_product(dz_ptr, &_z_rw_values[k * N_this], N_this) + 
                          simd::dot_product(dr_ptr, &_r_rw_values[k * N_this], N_this);
      }

      double* dx_t = &workspace.dx_matrix[(b_idx * num_time_steps + t) * N_prev];
      for (size_t k = 0; k < N_prev; ++k)
      {
        dx_t[k] = simd::dot_product(dz_ptr, &_z_w_values[k * N_this], N_this) + 
                  simd::dot_product(dr_ptr, &_r_w_values[k * N_this], N_this) + 
                  simd::dot_product(dh_hat_ptr, &_w_values[k * N_this], N_this);
      }

      double* dest = &rnn_grad_ptr_all[(b_idx * num_time_steps + t) * GateCount * N_this];
      std::copy(dh_hat_ptr, dh_hat_ptr + N_this, dest);
      std::copy(dz_ptr, dz_ptr + N_this, dest + N_this);
      std::copy(dr_ptr, dr_ptr + N_this, dest + 2 * N_this); // Gate 3 (Reset)
    }
  }

  for (size_t b = start; b < end; ++b)
  {
    const size_t b_idx = b - start;
    const double* dX_src = &workspace.dx_matrix[b_idx * num_time_steps * N_prev];
    batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), std::vector<double>(dX_src, dX_src + num_time_steps * N_prev));

    const double* gates_src = &workspace.rnn_grad_matrix[b_idx * num_time_steps * GateCount * N_this];
    batch_gradients_and_outputs[b].set_rnn_gate_gradients(get_layer_index(), std::vector<double>(gates_src, gates_src + num_time_steps * GateCount * N_this));
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

  // Launch threads for each batch chunk
  if (num_threads <= 1)
  {
    auto& workspace = get_workspace(0);
    calculate_bptt_batch_chunk(0, batch_size, batch_gradients_and_outputs, next_layer, batch_next_grad_matrix, batch_hidden_states, bptt_max_ticks, workspace, _rw_values_T, _z_rw_values_T, _r_rw_values_T);
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
  if (N_this == 0 || batch_size == 0) return;

  // Use a local FFLayer as a proxy for an identity connection.
  // This avoids duplicating the complex BPTT logic while correctly handling the MultiOutputLayer trunk gradients.
  FFLayer proxy(0, N_this, N_this, 0.0, Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, false, 0.0);
  std::vector<double> id(static_cast<size_t>(N_this) * N_this, 0.0);
  for (unsigned i = 0; i < N_this; ++i) id[i * N_this + i] = 1.0;
  proxy.set_w_values(id);

  calculate_hidden_gradients(batch_gradients_and_outputs, proxy, batch_output_gradients, batch_hidden_states, batch_size, bptt_max_ticks);
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
  const int active_ticks = t_start - t_end + 1;

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  std::vector<std::vector<double>> thread_w_grads(num_threads, std::vector<double>(_w_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_rw_grads(num_threads, std::vector<double>(_rw_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_z_w_grads(num_threads, std::vector<double>(_z_w_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_z_rw_grads(num_threads, std::vector<double>(_z_rw_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_r_w_grads(num_threads, std::vector<double>(_r_w_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_r_rw_grads(num_threads, std::vector<double>(_r_rw_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_b_grads(num_threads, std::vector<double>(has_bias() ? num_outputs : 0, 0.0));
  std::vector<std::vector<double>> thread_z_b_grads(num_threads, std::vector<double>(has_bias() ? num_outputs : 0, 0.0));
  std::vector<std::vector<double>> thread_r_b_grads(num_threads, std::vector<double>(has_bias() ? num_outputs : 0, 0.0));

  auto run_chunk = [&](size_t start, size_t end, size_t thread_idx)
  {
    auto& local_w_grads = thread_w_grads[thread_idx];
    auto& local_rw_grads = thread_rw_grads[thread_idx];
    auto& local_z_w_grads = thread_z_w_grads[thread_idx];
    auto& local_z_rw_grads = thread_z_rw_grads[thread_idx];
    auto& local_r_w_grads = thread_r_w_grads[thread_idx];
    auto& local_r_rw_grads = thread_r_rw_grads[thread_idx];
    auto& local_b_grads = thread_b_grads[thread_idx];
    auto& local_z_b_grads = thread_z_b_grads[thread_idx];
    auto& local_r_b_grads = thread_r_b_grads[thread_idx];

    for (size_t b = start; b < end; ++b)
    {
      const auto& rnn_grads = batch_gradients_and_outputs[b].get_rnn_gate_gradients(get_layer_index());
      const auto& prev_outputs_rnn = batch_gradients_and_outputs[b].get_rnn_outputs(previous_layer.get_layer_index());
      const auto& prev_outputs_std = batch_gradients_and_outputs[b].get_outputs(previous_layer.get_layer_index());
      const auto& prev_outputs = !prev_outputs_rnn.empty() ? prev_outputs_rnn : prev_outputs_std;

      if (rnn_grads.size() != static_cast<size_t>(num_time_steps) * GateCount * num_outputs) continue;

      for (int t = t_start; t >= t_end; --t)
      {
        const size_t base_idx = t * GateCount * num_outputs;
        const double* d_h_hat_ptr = &rnn_grads[base_idx];
        const double* d_z_ptr = &rnn_grads[base_idx + num_outputs];
        const double* d_r_ptr = &rnn_grads[base_idx + 2 * num_outputs];

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
        for (unsigned j = 0; j < num_outputs; ++j)
        {
          const double gh = d_h_hat_ptr[j];
          const double gz = d_z_ptr[j];
          const double gr = d_r_ptr[j];

          if (has_bias())
          {
            local_b_grads[j] += gh;
            local_z_b_grads[j] += gz;
            local_r_b_grads[j] += gr;
          }

          if (prev_input_ptr)
          {
            for (unsigned i = 0; i < num_inputs; ++i)
            {
              double x = prev_input_ptr[i];
              if (std::abs(x) < 1e-15)
              {
                continue;
              }
              local_w_grads[i * num_outputs + j] += gh * x;
              local_z_w_grads[i * num_outputs + j] += gz * x;
              local_r_w_grads[i * num_outputs + j] += gr * x;
            }
          }

          if (prev_hidden_ptr)
          {
            for (unsigned k = 0; k < num_outputs; ++k)
            {
              double h_prev = prev_hidden_ptr[k];
              if (std::abs(h_prev) < 1e-15)
              {
                continue;
              }
              double r_val = packed[num_outputs + k]; 
              local_rw_grads[k * num_outputs + j] += gh * (r_val * h_prev);
              local_z_rw_grads[k * num_outputs + j] += gz * h_prev;
              local_r_rw_grads[k * num_outputs + j] += gr * h_prev;
            }
          }
        }
      }
    }
  };

  if (num_threads <= 1)
  {
    run_chunk(0, batch_size, 0);
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
        _task_queue_pool->enqueue([start, end, t, &run_chunk]() { 
          run_chunk(start, end, t); 
        });
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  // Merge
  zero_gradients();
  for (unsigned int t = 0; t < num_threads; ++t)
  {
    for (size_t i = 0; i < _w_grads.size(); ++i)
    {
      _w_grads[i] += thread_w_grads[t][i];
      _z_w_grads[i] += thread_z_w_grads[t][i];
      _r_w_grads[i] += thread_r_w_grads[t][i];
    }
    for (size_t i = 0; i < _rw_grads.size(); ++i)
    {
      _rw_grads[i] += thread_rw_grads[t][i];
      _z_rw_grads[i] += thread_z_rw_grads[t][i];
      _r_rw_grads[i] += thread_r_rw_grads[t][i];
    }
    if (has_bias())
    {
      for (size_t i = 0; i < _b_grads.size(); ++i)
      {
        _b_grads[i] += thread_b_grads[t][i];
        _z_b_grads[i] += thread_z_b_grads[t][i];
        _r_b_grads[i] += thread_r_b_grads[t][i];
      }
    }
  }

  const double denom = static_cast<double>(batch_size);
  
  const auto normalize = [&denom](std::vector<double>& grads)
  {
    for (double& g : grads) g /= denom;
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
  norm_sq += std::inner_product(_w_grads.begin(), _w_grads.end(), _w_grads.begin(), 0.0);
  norm_sq += std::inner_product(_b_grads.begin(), _b_grads.end(), _b_grads.begin(), 0.0);
  norm_sq += std::inner_product(_rw_grads.begin(), _rw_grads.end(), _rw_grads.begin(), 0.0);
  norm_sq += std::inner_product(_z_w_grads.begin(), _z_w_grads.end(), _z_w_grads.begin(), 0.0);
  norm_sq += std::inner_product(_z_rw_grads.begin(), _z_rw_grads.end(), _z_rw_grads.begin(), 0.0);
  norm_sq += std::inner_product(_z_b_grads.begin(), _z_b_grads.end(), _z_b_grads.begin(), 0.0);
  norm_sq += std::inner_product(_r_w_grads.begin(), _r_w_grads.end(), _r_w_grads.begin(), 0.0);
  norm_sq += std::inner_product(_r_rw_grads.begin(), _r_rw_grads.end(), _r_rw_grads.begin(), 0.0);
  norm_sq += std::inner_product(_r_b_grads.begin(), _r_b_grads.end(), _r_b_grads.begin(), 0.0);

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

  zero_gradients();
}

void GRURNNLayer::cache_recurrent_weights()
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t n = get_number_neurons();
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
