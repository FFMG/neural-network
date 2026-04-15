#include "./libraries/instrumentor.h"
#include "grurnnlayer.h"
#include "logger.h"
#include <immintrin.h>

GRURNNLayer::GRURNNLayer(
  unsigned layer_index,
  unsigned num_neurons_in_previous_layer, 
  unsigned num_neurons_in_this_layer, 
  double weight_decay,
  LayerType layer_type, 
  const activation& activation_method,
  const OptimiserType& optimiser_type, 
  int residual_layer_number,
  double dropout_rate,
  ResidualProjector* residual_projector,
  int number_of_threads,
  bool has_bias
  ) :
  GRURNNLayer(
    layer_index,
    num_neurons_in_previous_layer,
    num_neurons_in_this_layer,
    std::vector<double>(static_cast<size_t>(num_neurons_in_previous_layer) * num_neurons_in_this_layer, weight_decay),
    layer_type,
    activation_method,
    optimiser_type,
    residual_layer_number,
    dropout_rate,
    residual_projector,
    number_of_threads,
    has_bias
  )
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
}

GRURNNLayer::GRURNNLayer(
  unsigned layer_index,
  unsigned num_neurons_in_previous_layer,
  unsigned num_neurons_in_this_layer,
  const std::vector<double>& weight_decays,
  LayerType layer_type,
  const activation& activation_method,
  const OptimiserType& optimiser_type,
  int residual_layer_number,
  double dropout_rate,
  ResidualProjector* residual_projector,
  int number_of_threads,
  bool has_bias
) :
  Layer(
    layer_index,
    layer_type,
    activation_method,
    optimiser_type,
    residual_layer_number,
    num_neurons_in_previous_layer,
    num_neurons_in_this_layer,
    create_neurons(dropout_rate, num_neurons_in_this_layer),
    has_bias,
    weight_decays,
    residual_projector,
    number_of_threads
  )
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  //  Note: we use the same weight decay for all GRU gates.
  initialize_recurrent_weights(weight_decays.empty() ? 0.0 : weight_decays[0]);
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
  _r_b_decays(std::move(src._r_b_decays))
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
}

GRURNNLayer::GRURNNLayer(
  unsigned layer_index,
  const LayerType layer_type,
  const activation& activation_method,
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
  int number_of_threads
) noexcept :
  Layer(
    layer_index,
    layer_type,
    activation_method,
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
  }
  return *this;
}

GRURNNLayer::~GRURNNLayer()
{
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
  values.clear();
  values.reserve(num_neurons);
  for (unsigned i = 0; i < num_neurons; ++i)
  {
    values.emplace_back(get_activation().weight_initialization());
  }
  grads.assign(num_neurons, 0.0);
  velocities.assign(num_neurons, 0.0);
  m1.assign(num_neurons, 0.0);
  m2.assign(num_neurons, 0.0);
  timesteps.assign(num_neurons, 0);
  decays.assign(num_neurons, 0.0); // No decay for biases
}

void GRURNNLayer::initialize_recurrent_weights(double weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const auto num_neurons = get_number_neurons();
  const auto num_inputs = get_number_input_neurons();

  const size_t num_rec_weights = static_cast<size_t>(num_neurons) * num_neurons;
  const size_t num_inp_weights = static_cast<size_t>(num_inputs) * num_neurons;

  // --- Helper Lambda for Weight Initialization ---
  auto init_weights = [&](std::vector<double>& values, std::vector<double>& grads, 
                          std::vector<double>& velocities, std::vector<double>& m1, 
                          std::vector<double>& m2, std::vector<long long>& timesteps, 
                          std::vector<double>& decays, size_t size, bool is_input)
  {
    values.resize(size);
    // Use the same initialization as the base layer
    // auto initial_weights = get_activation().weight_initialization((int)num_neurons, (int)(is_input ? num_inputs : num_neurons));
    for (size_t i = 0; i < size; ++i) {
      values[i] = get_activation().weight_initialization();// initial_weights[i % initial_weights.size()]; // Safety mod, though size should match
    }
    grads.assign(size, 0.0);
    velocities.assign(size, 0.0);
    m1.assign(size, 0.0);
    m2.assign(size, 0.0);
    timesteps.assign(size, 0);
    decays.assign(size, weight_decay);
  };

  // 1. Candidate State Recurrent Weights (using existing member)
  init_weights(_rw_values, _rw_grads, _rw_velocities, _rw_m1, _rw_m2, _rw_timesteps, _rw_decays, num_rec_weights, false);

  // 2. Update Gate (z)
  if (num_inputs > 0) 
  {
    init_weights(_z_w_values, _z_w_grads, _z_w_velocities, _z_w_m1, _z_w_m2, _z_w_timesteps, _z_w_decays, num_inp_weights, true);
  }
  init_weights(_z_rw_values, _z_rw_grads, _z_rw_velocities, _z_rw_m1, _z_rw_m2, _z_rw_timesteps, _z_rw_decays, num_rec_weights, false);
  if (has_bias()) 
  {
    init_bias(_z_b_values, _z_b_grads, _z_b_velocities, _z_b_m1, _z_b_m2, _z_b_timesteps, _z_b_decays);
  }

  // 3. Reset Gate (r)
  if (num_inputs > 0) 
  {
    init_weights(_r_w_values, _r_w_grads, _r_w_velocities, _r_w_m1, _r_w_m2, _r_w_timesteps, _r_w_decays, num_inp_weights, true);
  }
  init_weights(_r_rw_values, _r_rw_grads, _r_rw_velocities, _r_rw_m1, _r_rw_m2, _r_rw_timesteps, _r_rw_decays, num_rec_weights, false);
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
  if (batch_size == 0) return;

  const size_t N_prev = previous_layer.get_number_neurons();
  const size_t N_this = get_number_neurons();

  // 1. Flatten inputs [BatchSize x T x N_prev]
  std::vector<double> flattened_batch_inputs;
  size_t num_time_steps = 0;

  const unsigned prev_layer_index = previous_layer.get_layer_index();
  for (size_t b = 0; b < batch_size; ++b)
  {
      const std::vector<double> rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
      if (!rnn_in.empty())
      {
          const size_t t = rnn_in.size() / N_prev;
          if (num_time_steps == 0) {
              num_time_steps = t;
              flattened_batch_inputs.resize(batch_size * num_time_steps * N_prev);
          }
          std::copy(rnn_in.begin(), rnn_in.end(), flattened_batch_inputs.begin() + b * num_time_steps * N_prev);
      }
      else
      {
          const std::vector<double> std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
          if (std_in.size() == N_prev)
          {
              if (num_time_steps == 0) {
                  num_time_steps = 1;
                  flattened_batch_inputs.resize(batch_size * num_time_steps * N_prev);
              }
              for (size_t t = 0; t < num_time_steps; ++t)
                  std::copy(std_in.begin(), std_in.end(), flattened_batch_inputs.begin() + (b * num_time_steps + t) * N_prev);
          }
      }
  }

  if (num_time_steps == 0) return;

  // 2. Output sequence buffer
  std::vector<double> batch_output_sequences(batch_size * num_time_steps * N_this, 0.0);

  auto run_forward_pass = [&](size_t start, size_t end)
  {
    std::vector<double> z_pre(N_this), r_pre(N_this), h_hat_pre(N_this);
    std::vector<double> packed_bptt_states(3 * N_this);

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
          std::copy(get_b_values().begin(), get_b_values().end(), h_hat_pre.begin());
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
                    if (gated_h == 0.0) continue;
                    for (size_t j = j0; j < j_limit; ++j) h_hat_pre[j] += gated_h * U_h[i * N_this + j];
                }
            }
        }

        // f. Residuals and Candidate Activation
        if (!batch_residual_output_values.empty() && batch_residual_output_values[b].size() == N_this)
        {
            for (size_t j = 0; j < N_this; ++j) h_hat_pre[j] += batch_residual_output_values[b][j];
        }

        std::vector<double> h_hat_vec = h_hat_pre;
        get_activation().activate(h_hat_vec.data(), h_hat_vec.data() + N_this);

        // g. Final State Update
        for (size_t j = 0; j < N_this; ++j)
        {
          packed_bptt_states[2 * N_this + j] = h_hat_pre[j];
          double h_hat = h_hat_vec[j];
          if (is_training && get_neuron((unsigned)j).is_dropout())
          {
              const auto& neuron = get_neuron((unsigned)j);
              if (neuron.must_randomly_drop()) h_hat = 0.0;
              else h_hat /= (1.0 - neuron.get_dropout_rate());
          }
          current_h[j] = (1.0 - packed_bptt_states[j]) * prev_h[j] + packed_bptt_states[j] * h_hat;
          batch_output_sequences[(b * num_time_steps + t) * N_this + j] = current_h[j];
        }

        batch_hidden_states[b].at(get_layer_index())[t].set_pre_activation_sums(packed_bptt_states);
        batch_hidden_states[b].at(get_layer_index())[t].set_hidden_state_values(current_h);
        prev_h = current_h;
      }
    }
  };

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads <= 1)
  {
    run_forward_pass(0, batch_size);
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
        _task_queue_pool->enqueue([run_forward_pass, start, end, this]()
          {
            run_forward_pass(start, end);
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

void GRURNNLayer::calculate_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  Logger::panic("GRURNNLayer: Trying to calculate output gradient with a non output layer!");
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
  const BPTTWorkspace::AlignedVector& rw_values_T,
  const BPTTWorkspace::AlignedVector& z_rw_values_T,
  const BPTTWorkspace::AlignedVector& r_rw_values_T) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_next = next_layer.get_number_neurons();
  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  const int t_start = static_cast<int>(num_time_steps) - 1;
  int t_end = (bptt_max_ticks > 0) ? std::max(0, t_start - bptt_max_ticks + 1) : 0;

  // Use workspace instead of local allocations to avoid repeated memory management
  workspace.grad_from_next_all_t.assign((end - start) * num_time_steps * N_this, 0.0);
  workspace.d_next_h.assign((end - start) * N_this, 0.0);
  workspace.rnn_grad_matrix.assign((end - start) * num_time_steps * 3 * N_this, 0.0);

  // Precompute gradients from next layer (all time steps) using tiling for cache locality
  constexpr size_t BLOCK_SIZE = 64;
  double* grad_next_all_ptr = workspace.grad_from_next_all_t.data();

  for (size_t b_idx0 = 0; b_idx0 < end - start; b_idx0 += BLOCK_SIZE)
  {
    size_t b_idx_limit = std::min(b_idx0 + BLOCK_SIZE, end - start);
    for (int t = t_start; t >= t_end; --t)
    {
      for (size_t i0 = 0; i0 < N_this; i0 += BLOCK_SIZE)
      {
        size_t i_limit = std::min(i0 + BLOCK_SIZE, N_this);
        for (size_t b_idx = b_idx0; b_idx < b_idx_limit; ++b_idx)
        {
          size_t b = start + b_idx;
          const auto& next_grad_matrix = batch_next_grad_matrix[b];
          const bool next_grad_is_seq = (next_grad_matrix.size() == N_next * num_time_steps);

          const double* next_grad_ptr = nullptr;
          if (next_grad_is_seq)
          {
            next_grad_ptr = &next_grad_matrix[t * N_next];
          }
          else if (t == t_start && next_grad_matrix.size() == N_next)
          {
            next_grad_ptr = next_grad_matrix.data();
          }

          if (next_grad_ptr)
          {
            double* dest_ptr = &grad_next_all_ptr[(b_idx * num_time_steps + t) * N_this];
            for (size_t i = i0; i < i_limit; ++i)
            {
              const double* next_w_row = next_layer.get_w_values().data() + i * N_next;
              
              // Vectorized Dot Product for Step 0
              __m256d sum_vec = _mm256_setzero_pd();
              size_t k = 0;
              for (; k + 3 < N_next; k += 4)
              {
                __m256d g = _mm256_loadu_pd(&next_grad_ptr[k]);
                __m256d w = _mm256_loadu_pd(&next_w_row[k]);
                sum_vec = _mm256_fmadd_pd(g, w, sum_vec);
              }
              
              double buffer[4];
              _mm256_storeu_pd(buffer, sum_vec);
              double sum = buffer[0] + buffer[1] + buffer[2] + buffer[3];
              for (; k < N_next; ++k) sum += next_grad_ptr[k] * next_w_row[k];
              
              dest_ptr[i] += sum;
            }
          }
        }
      }
    }
  }

  // Prep chunk-level buffers from workspace
  workspace.chunk_dz.resize((end - start) * N_this);
  workspace.chunk_dr.resize((end - start) * N_this);
  workspace.chunk_dh_hat.resize((end - start) * N_this);
  workspace.chunk_dh_prev_accum.resize((end - start) * N_this);
  workspace.h_hat_vals.resize(N_this);

  double* dz_ptr_all = workspace.chunk_dz.data();
  double* dr_ptr_all = workspace.chunk_dr.data();
  double* dh_hat_ptr_all = workspace.chunk_dh_hat.data();
  double* dh_prev_accum_ptr_all = workspace.chunk_dh_prev_accum.data();
  double* d_next_h_ptr_base = workspace.d_next_h.data();
  double* rnn_grad_ptr_all = workspace.rnn_grad_matrix.data();

  // BPTT Loop
  for (int t = t_start; t >= t_end; --t)
  {
    std::fill(workspace.chunk_dh_prev_accum.begin(), workspace.chunk_dh_prev_accum.end(), 0.0);

    // Step 1: dz, dh_hat (Vectorized)
    for (size_t b_idx = 0; b_idx < end - start; ++b_idx)
    {
      size_t b = start + b_idx;
      const size_t t_offset = (b_idx * num_time_steps + t) * N_this;
      const auto& packed_states = batch_hidden_states[b].at(get_layer_index())[t].get_pre_activation_sums();

      const double* z_vals = &packed_states[0];
      const double* h_hat_pre_vals = &packed_states[2 * N_this];
      const double* h_prev_vals = (t > 0) ? batch_hidden_states[b].at(get_layer_index())[t - 1].get_hidden_state_values().data() : nullptr;

      const double* grad_next_ptr = &grad_next_all_ptr[t_offset];
      const double* d_next_h_ptr = &d_next_h_ptr_base[b_idx * N_this];

      std::copy(h_hat_pre_vals, h_hat_pre_vals + N_this, workspace.h_hat_vals.begin());
      get_activation().activate(workspace.h_hat_vals.data(), workspace.h_hat_vals.data() + N_this);

      const double* h_hat_ptr = workspace.h_hat_vals.data();

      size_t j = 0;
      __m256d one = _mm256_set1_pd(1.0);
      for (; j + 3 < N_this; j += 4)
      {
        __m256d dh = _mm256_add_pd(_mm256_loadu_pd(&grad_next_ptr[j]), _mm256_loadu_pd(&d_next_h_ptr[j]));
        __m256d z = _mm256_loadu_pd(&z_vals[j]);
        __m256d h_hat = _mm256_loadu_pd(&h_hat_ptr[j]);
        __m256d h_prev = h_prev_vals ? _mm256_loadu_pd(&h_prev_vals[j]) : _mm256_setzero_pd();

        __m256d d_z = _mm256_mul_pd(dh, _mm256_sub_pd(h_hat, h_prev));
        __m256d d_z_pre = _mm256_mul_pd(d_z, _mm256_mul_pd(z, _mm256_sub_pd(one, z)));

        __m256d d_h_hat = _mm256_mul_pd(dh, z);
        
        // Scalar fallback for activation derivative (hard to vectorize virtual call)
        double derivatives[4];
        for(int k=0; k<4; ++k) derivatives[k] = get_activation().activate_derivative(h_hat_pre_vals[j+k]);
        __m256d d_h_hat_pre = _mm256_mul_pd(d_h_hat, _mm256_loadu_pd(derivatives));
        
        __m256d d_h_prev_direct = _mm256_mul_pd(dh, _mm256_sub_pd(one, z));

        _mm256_storeu_pd(&dz_ptr_all[b_idx * N_this + j], d_z_pre);
        _mm256_storeu_pd(&dh_hat_ptr_all[b_idx * N_this + j], d_h_hat_pre);
        _mm256_storeu_pd(&dh_prev_accum_ptr_all[b_idx * N_this + j], d_h_prev_direct);
      }

      for (; j < N_this; ++j)
      {
        double dh = grad_next_ptr[j] + d_next_h_ptr[j];
        double z = z_vals[j];
        double h_hat = h_hat_ptr[j];
        double h_prev = (h_prev_vals) ? h_prev_vals[j] : 0.0;

        double d_z = dh * (h_hat - h_prev);
        double d_z_pre = d_z * z * (1.0 - z);

        double d_h_hat = dh * z;
        double d_h_hat_pre = d_h_hat * get_activation().activate_derivative(h_hat_pre_vals[j]);
        double d_h_prev_direct = dh * (1.0 - z);

        dz_ptr_all[b_idx * N_this + j] = d_z_pre;
        dh_hat_ptr_all[b_idx * N_this + j] = d_h_hat_pre;
        dh_prev_accum_ptr_all[b_idx * N_this + j] = d_h_prev_direct;
      }
    }

    // Step 2: temp_Uh (dL/dr part) hoisted from the time loop (Vectorized Dot Product)
    // Using transposed candidate recurrent weights rw_values_T
    workspace.temp_Uh_T_dh_hat.assign((end - start) * N_this, 0.0);
    double* temp_Uh_ptr_all = workspace.temp_Uh_T_dh_hat.data();
    const double* rw_values_T_ptr = rw_values_T.data();

    for (size_t b_idx0 = 0; b_idx0 < end - start; b_idx0 += BLOCK_SIZE)
    {
      size_t b_idx_limit = std::min(b_idx0 + BLOCK_SIZE, end - start);
      for (size_t i0 = 0; i0 < N_this; i0 += BLOCK_SIZE)
      {
        size_t i_limit = std::min(i0 + BLOCK_SIZE, N_this);
        for (size_t j0 = 0; j0 < N_this; j0 += BLOCK_SIZE)
        {
          size_t j_limit = std::min(j0 + BLOCK_SIZE, N_this);

          for (size_t i = i0; i < i_limit; ++i)
          {
            const double* w_row = &rw_values_T_ptr[i * N_this];
            for (size_t b_idx = b_idx0; b_idx < b_idx_limit; ++b_idx)
            {
              const double* dh_hat_ptr = &dh_hat_ptr_all[b_idx * N_this];
              
              __m256d sum_vec = _mm256_setzero_pd();
              size_t j = j0;
              for (; j + 3 < j_limit; j += 4)
              {
                sum_vec = _mm256_fmadd_pd(_mm256_loadu_pd(&dh_hat_ptr[j]), _mm256_loadu_pd(&w_row[j]), sum_vec);
              }
              double buffer[4];
              _mm256_storeu_pd(buffer, sum_vec);
              double sum = buffer[0] + buffer[1] + buffer[2] + buffer[3];
              for (; j < j_limit; ++j) sum += dh_hat_ptr[j] * w_row[j];
              
              temp_Uh_ptr_all[b_idx * N_this + i] += sum;
            }
          }
        }
      }
    }

    // Step 3: dr and propagate using tiling and cache-friendly access (Vectorized Dot Product)
    for (size_t b_idx0 = 0; b_idx0 < end - start; b_idx0 += BLOCK_SIZE)
    {
      size_t b_idx_limit = std::min(b_idx0 + BLOCK_SIZE, end - start);
      
      // 3a. dr calculation (remains simple, but tiled for consistency)
      for (size_t b_idx = b_idx0; b_idx < b_idx_limit; ++b_idx)
      {
        size_t b = start + b_idx;
        const auto& packed_states = batch_hidden_states[b].at(get_layer_index())[t].get_pre_activation_sums();
        const double* r_vals = &packed_states[N_this];
        const double* h_prev_vals = (t > 0) ? batch_hidden_states[b].at(get_layer_index())[t - 1].get_hidden_state_values().data() : nullptr;
        const double* temp_Uh_ptr = &temp_Uh_ptr_all[b_idx * N_this];

        size_t i = 0;
        for (; i + 3 < N_this; i += 4)
        {
          __m256d grad_rh = _mm256_loadu_pd(&temp_Uh_ptr[i]);
          __m256d h_prev = h_prev_vals ? _mm256_loadu_pd(&h_prev_vals[i]) : _mm256_setzero_pd();
          __m256d r = _mm256_loadu_pd(&r_vals[i]);
          
          __m256d d_r = _mm256_mul_pd(grad_rh, h_prev);
          __m256d d_r_pre = _mm256_mul_pd(d_r, _mm256_mul_pd(r, _mm256_sub_pd(_mm256_set1_pd(1.0), r)));
          _mm256_storeu_pd(&dr_ptr_all[b_idx * N_this + i], d_r_pre);
          
          __m256d dh_prev = _mm256_mul_pd(grad_rh, r);
          __m256d current_dh_prev = _mm256_loadu_pd(&dh_prev_accum_ptr_all[b_idx * N_this + i]);
          _mm256_storeu_pd(&dh_prev_accum_ptr_all[b_idx * N_this + i], _mm256_add_pd(current_dh_prev, dh_prev));
        }

        for (; i < N_this; ++i)
        {
          double grad_rh = temp_Uh_ptr[i];
          double h_prev = (h_prev_vals) ? h_prev_vals[i] : 0.0;
          double r = r_vals[i];

          double d_r = grad_rh * h_prev;
          double d_r_pre = d_r * r * (1.0 - r);
          dr_ptr_all[b_idx * N_this + i] = d_r_pre;
          dh_prev_accum_ptr_all[b_idx * N_this + i] += grad_rh * r;
        }
      }

      // 3b. Propagate U_z and U_r using tiling (Vectorized Dot Product)
      // Using transposed gate recurrent weights
      const double* z_rw_values_T_ptr = z_rw_values_T.data();
      const double* r_rw_values_T_ptr = r_rw_values_T.data();

      for (size_t i0 = 0; i0 < N_this; i0 += BLOCK_SIZE)
      {
        size_t i_limit = std::min(i0 + BLOCK_SIZE, N_this);
        for (size_t j0 = 0; j0 < N_this; j0 += BLOCK_SIZE)
        {
          size_t j_limit = std::min(j0 + BLOCK_SIZE, N_this);

          for (size_t i = i0; i < i_limit; ++i)
          {
            const double* wz_row = &z_rw_values_T_ptr[i * N_this];
            const double* wr_row = &r_rw_values_T_ptr[i * N_this];
            for (size_t b_idx = b_idx0; b_idx < b_idx_limit; ++b_idx)
            {
              const double* dz_ptr = &dz_ptr_all[b_idx * N_this];
              const double* dr_ptr = &dr_ptr_all[b_idx * N_this];
              
              __m256d sum_vec = _mm256_setzero_pd();
              size_t j = j0;
              for (; j + 3 < j_limit; j += 4)
              {
                sum_vec = _mm256_fmadd_pd(_mm256_loadu_pd(&dz_ptr[j]), _mm256_loadu_pd(&wz_row[j]), sum_vec);
                sum_vec = _mm256_fmadd_pd(_mm256_loadu_pd(&dr_ptr[j]), _mm256_loadu_pd(&wr_row[j]), sum_vec);
              }
              double buffer[4];
              _mm256_storeu_pd(buffer, sum_vec);
              double sum = buffer[0] + buffer[1] + buffer[2] + buffer[3];
              for (; j < j_limit; ++j) sum += dz_ptr[j] * wz_row[j] + dr_ptr[j] * wr_row[j];
              
              dh_prev_accum_ptr_all[b_idx * N_this + i] += sum;
            }
          }
        }
      }
    }

    // Store state for next iter
    std::copy(workspace.chunk_dh_prev_accum.begin(), workspace.chunk_dh_prev_accum.end(), workspace.d_next_h.begin());

    // Store gradients to output matrix (needed for previous layer)
    // Packed: [d_h_hat, d_z, d_r]
    for (size_t b_idx = 0; b_idx < end - start; ++b_idx)
    {
      size_t base_idx = (b_idx * num_time_steps + t) * 3 * N_this;
      double* dest = &rnn_grad_ptr_all[base_idx];
      const double* src_h_hat = &dh_hat_ptr_all[b_idx * N_this];
      const double* src_dz = &dz_ptr_all[b_idx * N_this];
      const double* src_dr = &dr_ptr_all[b_idx * N_this];

      size_t j = 0;
      for (; j + 3 < N_this; j += 4)
      {
        _mm256_storeu_pd(&dest[j], _mm256_loadu_pd(&src_h_hat[j]));
        _mm256_storeu_pd(&dest[N_this + j], _mm256_loadu_pd(&src_dz[j]));
        _mm256_storeu_pd(&dest[2 * N_this + j], _mm256_loadu_pd(&src_dr[j]));
      }
      for (; j < N_this; ++j)
      {
        dest[j] = src_h_hat[j];
        dest[N_this + j] = src_dz[j];
        dest[2 * N_this + j] = src_dr[j];
      }
    }
  } // End BPTT Loop

  // Copy results - optimized to write directly into GradientsAndOutputs buffers
  const size_t grad_size = num_time_steps * 3 * N_this;
  for (size_t b_idx = 0; b_idx < end - start; ++b_idx)
  {
    size_t b = start + b_idx;
    
    // 1. Write RNN gradients directly into GradientsAndOutputs internal buffer
    double* dest = batch_gradients_and_outputs[b].get_rnn_gradients_raw(get_layer_index(), grad_size);
    const double* src = &workspace.rnn_grad_matrix[b_idx * grad_size];
    std::copy(src, src + grad_size, dest);
    
    // 2. Efficiently zero out standard gradients for this layer index
    double* std_grads = batch_gradients_and_outputs[b].get_gradients_raw(get_layer_index());
    if (std_grads)
    {
      std::fill(std_grads, std_grads + N_this, 0.0);
    }
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
  if (batch_size == 0) return;

  const size_t N_this = get_number_neurons();

  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  if (num_time_steps == 0 || N_this == 0) return;

  const auto& num_threads = _task_queue_pool->get_number_of_threads();

  // Transpose recurrent weights once per batch for Step 2 and 3b cache-friendly access
  // These are now local to avoid thread contention.
  BPTTWorkspace::AlignedVector rw_values_T(N_this * N_this);
  BPTTWorkspace::AlignedVector z_rw_values_T(N_this * N_this);
  BPTTWorkspace::AlignedVector r_rw_values_T(N_this * N_this);

  for (size_t i = 0; i < N_this; ++i)
  {
    for (size_t j = 0; j < N_this; ++j)
    {
      rw_values_T[j * N_this + i] = _rw_values[i * N_this + j];
      z_rw_values_T[j * N_this + i] = _z_rw_values[i * N_this + j];
      r_rw_values_T[j * N_this + i] = _r_rw_values[i * N_this + j];
    }
  }

  // Launch threads for each batch chunk
  if (num_threads <= 1)
  {
    BPTTWorkspace local_workspace;
    calculate_bptt_batch_chunk(0, batch_size, batch_gradients_and_outputs, next_layer, batch_next_grad_matrix, batch_hidden_states, bptt_max_ticks, local_workspace, rw_values_T, z_rw_values_T, r_rw_values_T);
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
        _task_queue_pool->enqueue([start, end, &batch_gradients_and_outputs, &next_layer, &batch_next_grad_matrix, &batch_hidden_states, bptt_max_ticks, &rw_values_T, &z_rw_values_T, &r_rw_values_T, this]()
          {
            BPTTWorkspace thread_local_workspace;
            calculate_bptt_batch_chunk(start, end, batch_gradients_and_outputs, next_layer, batch_next_grad_matrix, batch_hidden_states, bptt_max_ticks, thread_local_workspace, rw_values_T, z_rw_values_T, r_rw_values_T);
          });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
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

  // 1. Clear gradients
  std::fill(_w_grads.begin(), _w_grads.end(), 0.0);
  std::fill(_rw_grads.begin(), _rw_grads.end(), 0.0);
  if (has_bias())
  {
    std::fill(_b_grads.begin(), _b_grads.end(), 0.0);
  }

  std::fill(_z_w_grads.begin(), _z_w_grads.end(), 0.0);
  std::fill(_z_rw_grads.begin(), _z_rw_grads.end(), 0.0);
  if (has_bias())
  {
    std::fill(_z_b_grads.begin(), _z_b_grads.end(), 0.0);
  }

  std::fill(_r_w_grads.begin(), _r_w_grads.end(), 0.0);
  std::fill(_r_rw_grads.begin(), _r_rw_grads.end(), 0.0);
  if (has_bias())
  {
    std::fill(_r_b_grads.begin(), _r_b_grads.end(), 0.0);
  }

  const unsigned num_time_steps = (unsigned)hidden_states[0].at(get_layer_index()).size();
  const int t_start = static_cast<int>(num_time_steps) - 1;
  const int t_end = (bptt_max_ticks > 0) ? std::max(0, t_start - bptt_max_ticks + 1) : 0;
  const int active_ticks = t_start - t_end + 1;

  const double time_scale = (active_ticks > 0) ? static_cast<double>(active_ticks) : 1.0;
  
  // Normalization follows ElmanRNNLayer pattern:
  // - Input weights (W, z_w, r_w): normalized by (batch_size * active_ticks)
  // - Recurrent weights (RW, z_rw, r_rw): normalized by (batch_size * (active_ticks - 1))
  // This matches the number of times each weight is used in BPTT.
  const double denom = static_cast<double>(batch_size) * time_scale;

  // Use a thread-local accumulator pattern or just atomic accumulation?
  // Since we are running single-threaded per layer in update_weights usually (or layer-parallel),
  // we can just accumulate directly.
  for (unsigned b = 0; b < batch_size; ++b)
  {
    const auto& rnn_grads = batch_gradients_and_outputs[b].get_rnn_gradients(get_layer_index());
    const auto& prev_outputs_rnn = batch_gradients_and_outputs[b].get_rnn_outputs(previous_layer.get_layer_index());
    const auto& prev_outputs_std = batch_gradients_and_outputs[b].get_outputs(previous_layer.get_layer_index());
    const auto& prev_outputs = !prev_outputs_rnn.empty() ? prev_outputs_rnn : prev_outputs_std;

    // We expect rnn_grads to be packed: [T * 3 * N]
    // Layout at time t: [d_h_hat (N), d_z (N), d_r (N)]
    if (rnn_grads.size() != static_cast<size_t>(num_time_steps) * 3 * num_outputs) continue;

    for (int t = t_start; t >= t_end; --t)
    {
      const size_t base_idx = t * 3 * num_outputs;
      const double* d_h_hat_ptr = &rnn_grads[base_idx];
      const double* d_z_ptr = &rnn_grads[base_idx + num_outputs];
      const double* d_r_ptr = &rnn_grads[base_idx + 2 * num_outputs];

      const double* prev_input_ptr = nullptr;
      if (prev_outputs.size() == num_inputs)
      {
        prev_input_ptr = prev_outputs.data(); // Static input
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

      // 1. Input Weights Gradients (W * x)
      if (prev_input_ptr)
      {
        for (unsigned j = 0; j < num_outputs; ++j)
        {
          double dh = d_h_hat_ptr[j];
          double dz = d_z_ptr[j];
          double dr = d_r_ptr[j];

          for (unsigned i = 0; i < num_inputs; ++i)
          {
            double x = prev_input_ptr[i];
            _w_grads[i * num_outputs + j] += dh * x;
            _z_w_grads[i * num_outputs + j] += dz * x;
            _r_w_grads[i * num_outputs + j] += dr * x;
          }
        }
      }

      // 2. Recurrent Weights Gradients (U * h_{t-1})
      if (prev_hidden_ptr)
      {
        for (unsigned j = 0; j < num_outputs; ++j)
        {
          double dh = d_h_hat_ptr[j];
          double dz = d_z_ptr[j];
          double dr = d_r_ptr[j];
               
          // For h_hat, the recurrent input is actually (r_t * h_{t-1})
          // We need r_t for this sample/time.
          // It's stored in hidden_states[t] pre-activation sums, index N..2N (sigmoid applied? No, sums are pre-activations).
          // Wait, apply_stored_gradients needs gradients w.r.t weights.
          // dL/dU_h = dL/d_h_hat_pre * (r * h_{t-1})
          // dL/dU_z = dL/d_z_pre * h_{t-1}
          // dL/dU_r = dL/d_r_pre * h_{t-1}

          // Retrieve r_t (post-activation)
          // The hidden_state stores pre-activations.
          // batch_hidden_states[b].at(get_layer_index())[t].get_pre_activation_sums()[N + j] is r_pre?
          // Wait, in calculate_forward_feed:
          // packed_bptt_states[j] = z_val;          // Index 0..N-1
          // packed_bptt_states[N_this + j] = r_val; // Index N..2N-1
          // packed_bptt_states[2 * N_this + j] = h_hat_pre[j];
               
          const auto& packed = hidden_states[b].at(get_layer_index())[t].get_pre_activation_sums();
          double r_val = packed[num_outputs + j]; 
          // Note: packed stores ACTIVATED z and r values (0..1), but PRE-ACTIVATED h_hat.
               
          for (unsigned k = 0; k < num_outputs; ++k)
          {
            double h_prev = prev_hidden_ptr[k];
                   
            // Candidate State Recurrent: U_h * (r * h_prev)
            _rw_grads[k * num_outputs + j] += dh * (r_val * h_prev);

            // Update Gate Recurrent: U_z * h_prev
            _z_rw_grads[k * num_outputs + j] += dz * h_prev;

            // Reset Gate Recurrent: U_r * h_prev
            _r_rw_grads[k * num_outputs + j] += dr * h_prev;
          }
        }
      }

      // 3. Bias Gradients
      if (has_bias())
      {
        for (unsigned j = 0; j < num_outputs; ++j)
        {
          _b_grads[j] += d_h_hat_ptr[j];
          _z_b_grads[j] += d_z_ptr[j];
          _r_b_grads[j] += d_r_ptr[j];
        }
      }
    }
  }

  // Final Normalization
  auto normalize = [&](std::vector<double>& grads)
  {
    for (double& g : grads)
    {
      g /= denom;
    }
  };

  normalize(_w_grads);
  normalize(_z_w_grads);
  normalize(_r_w_grads);

  // Recurrent weights might typically be normalized differently if sequence length varies,
  // but here we use the same denom.
  // Note: ElmanRNNLayer normalizes recurrent weights by (batch_size * (active_ticks - 1))?
  // Let's check ElmanRNNLayer: 
  // const double time_denom_rec = (active_ticks > 1) ? static_cast<double>(active_ticks - 1) : 1.0;
  // for (double& grad : _rw_grads) grad /= (static_cast<double>(batch_size) * time_denom_rec);
  
  const double time_denom_rec = (active_ticks > 1) ? static_cast<double>(active_ticks - 1) : 1.0;
  const double denom_rec = static_cast<double>(batch_size) * time_denom_rec;
  
  auto normalize_rec = [&](std::vector<double>& grads)
    {
      for (double& g : grads)
      {
        g /= denom_rec;
      }
    };

  normalize_rec(_rw_grads);
  normalize_rec(_z_rw_grads);
  normalize_rec(_r_rw_grads);

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
  auto sum_sq = [&](const std::vector<double>& grads) 
  {
    for (const double grad : grads)
    {
      norm_sq += grad * grad;
    }
  };
    
  sum_sq(_w_grads);
  sum_sq(_b_grads);
  sum_sq(_rw_grads);
  sum_sq(_z_w_grads);
  sum_sq(_z_rw_grads);
  sum_sq(_z_b_grads);
  sum_sq(_r_w_grads);
  sum_sq(_r_rw_grads);
  sum_sq(_r_b_grads);

  return norm_sq;
}

void GRURNNLayer::apply_stored_gradients(double learning_rate, double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");

  const unsigned num_outputs = get_number_neurons();
  const unsigned num_inputs = get_number_input_neurons();

  // Iterate over all neurons
  for (unsigned j = 0; j < num_outputs; ++j)
  {
    // 1. Input-to-Hidden Weights
    for (unsigned i = 0; i < num_inputs; ++i)
    {
      unsigned idx = i * num_outputs + j;

      // A. Candidate State (Uses Base Layer storage)
      apply_weight_gradient(_w_grads[idx], learning_rate, false, idx, clipping_scale, _optimiser_type);

      // B. Update Gate (z)
      apply_update_to_weight(_z_w_values, _z_w_grads, _z_w_velocities, _z_w_m1, _z_w_m2, _z_w_timesteps, _z_w_decays, idx, _z_w_grads[idx], learning_rate, clipping_scale, _optimiser_type);

      // C. Reset Gate (r)
      apply_update_to_weight(_r_w_values, _r_w_grads, _r_w_velocities, _r_w_m1, _r_w_m2, _r_w_timesteps, _r_w_decays, idx, _r_w_grads[idx], learning_rate, clipping_scale, _optimiser_type);
    }

    // 2. Recurrent Weights (Hidden-to-Hidden)
    for (unsigned k = 0; k < num_outputs; ++k)
    {
       // Indexing for recurrent weights: k (from) * num_outputs + j (to)
       unsigned rec_idx = k * num_outputs + j;

       // A. Candidate State
       apply_update_to_weight(_rw_values, _rw_grads, _rw_velocities, _rw_m1, _rw_m2, _rw_timesteps, _rw_decays, rec_idx, _rw_grads[rec_idx], learning_rate, clipping_scale, _optimiser_type);

       // B. Update Gate (z)
       apply_update_to_weight(_z_rw_values, _z_rw_grads, _z_rw_velocities, _z_rw_m1, _z_rw_m2, _z_rw_timesteps, _z_rw_decays, rec_idx, _z_rw_grads[rec_idx], learning_rate, clipping_scale, _optimiser_type);

       // C. Reset Gate (r)
       apply_update_to_weight(_r_rw_values, _r_rw_grads, _r_rw_velocities, _r_rw_m1, _r_rw_m2, _r_rw_timesteps, _r_rw_decays, rec_idx, _r_rw_grads[rec_idx], learning_rate, clipping_scale, _optimiser_type);
    }

    // 3. Bias Weights
    if (has_bias())
    {
       // A. Candidate State
       apply_weight_gradient(_b_grads[j], learning_rate, true, j, clipping_scale, _optimiser_type);

       // B. Update Gate (z)
       apply_update_to_weight(_z_b_values, _z_b_grads, _z_b_velocities, _z_b_m1, _z_b_m2, _z_b_timesteps, _z_b_decays, j, _z_b_grads[j], learning_rate, clipping_scale, _optimiser_type);

       // C. Reset Gate (r)
       apply_update_to_weight(_r_b_values, _r_b_grads, _r_b_velocities, _r_b_m1, _r_b_m2, _r_b_timesteps, _r_b_decays, j, _r_b_grads[j], learning_rate, clipping_scale, _optimiser_type);
    }
  }
}
