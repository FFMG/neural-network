#include "./libraries/instrumentor.h"
#include "grurnnlayer.h"
#include "logger.h"

constexpr bool _has_bias_neuron = true;

GRURNNLayer::GRURNNLayer(
  unsigned layer_index,
  unsigned num_neurons_in_previous_layer, 
  unsigned num_neurons_in_this_layer, 
  double weight_decay,
  LayerType layer_type, 
  const activation::method& activation_method,
  const OptimiserType& optimiser_type, 
  int residual_layer_number,
  double dropout_rate,
  ResidualProjector* residual_projector
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
    _has_bias_neuron,
    weight_decay,
    residual_projector
  ),
  _task_queue_pool(nullptr)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  initialize_recurrent_weights(weight_decay);
  _task_queue_pool = new TaskQueuePool<void>();
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
  _r_b_decays(src._r_b_decays),
  _task_queue_pool(nullptr)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  if (src._task_queue_pool != nullptr)
  {
    _task_queue_pool = new TaskQueuePool<void>(src._task_queue_pool->get_number_of_threads());
  }
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
  _task_queue_pool(src._task_queue_pool)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  src._task_queue_pool = nullptr;
}

GRURNNLayer::GRURNNLayer(
  unsigned layer_index,
  const LayerType layer_type,
  const activation activation,
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
  const ResidualProjector* residual_projector
) noexcept :
  Layer(
    layer_index,
    layer_type,
    activation,
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
    residual_projector),
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
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  _task_queue_pool = new TaskQueuePool<void>();
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

    delete _task_queue_pool;
    _task_queue_pool = nullptr;
    if (src._task_queue_pool != nullptr)
    {
      _task_queue_pool = new TaskQueuePool<void>(src._task_queue_pool->get_number_of_threads());
    }
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

    delete _task_queue_pool;
    _task_queue_pool = src._task_queue_pool;
    src._task_queue_pool = nullptr;
  }
  return *this;
}

GRURNNLayer::~GRURNNLayer()
{
  delete _task_queue_pool;
}

void GRURNNLayer::initialize_recurrent_weights(double weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const auto num_neurons = get_number_output_neurons();
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
    auto initial_weights = get_activation().weight_initialization(num_neurons, is_input ? num_inputs : num_neurons);
    for (size_t i = 0; i < size; ++i) {
        values[i] = initial_weights[i % initial_weights.size()]; // Safety mod, though size should match
    }
    grads.assign(size, 0.0);
    velocities.assign(size, 0.0);
    m1.assign(size, 0.0);
    m2.assign(size, 0.0);
    timesteps.assign(size, 0);
    decays.assign(size, weight_decay);
  };

  // --- Helper Lambda for Bias Initialization ---
  auto init_bias = [&](std::vector<double>& values, std::vector<double>& grads, 
                       std::vector<double>& velocities, std::vector<double>& m1, 
                       std::vector<double>& m2, std::vector<long long>& timesteps, 
                       std::vector<double>& decays)
  {
      values = get_activation().weight_initialization(num_neurons, 1);
      grads.assign(num_neurons, 0.0);
      velocities.assign(num_neurons, 0.0);
      m1.assign(num_neurons, 0.0);
      m2.assign(num_neurons, 0.0);
      timesteps.assign(num_neurons, 0);
      decays.assign(num_neurons, 0.0); // No decay for biases
  };

  // 1. Candidate State Recurrent Weights (using existing member)
  init_weights(_rw_values, _rw_grads, _rw_velocities, _rw_m1, _rw_m2, _rw_timesteps, _rw_decays, num_rec_weights, false);

  // 2. Update Gate (z)
  if (num_inputs > 0) {
      init_weights(_z_w_values, _z_w_grads, _z_w_velocities, _z_w_m1, _z_w_m2, _z_w_timesteps, _z_w_decays, num_inp_weights, true);
  }
  init_weights(_z_rw_values, _z_rw_grads, _z_rw_velocities, _z_rw_m1, _z_rw_m2, _z_rw_timesteps, _z_rw_decays, num_rec_weights, false);
  if (has_bias()) {
      init_bias(_z_b_values, _z_b_grads, _z_b_velocities, _z_b_m1, _z_b_m2, _z_b_timesteps, _z_b_decays);
  }

  // 3. Reset Gate (r)
  if (num_inputs > 0) {
      init_weights(_r_w_values, _r_w_grads, _r_w_velocities, _r_w_m1, _r_w_m2, _r_w_timesteps, _r_w_decays, num_inp_weights, true);
  }
  init_weights(_r_rw_values, _r_rw_grads, _r_rw_velocities, _r_rw_m1, _r_rw_m2, _r_rw_timesteps, _r_rw_decays, num_rec_weights, false);
  if (has_bias()) {
      init_bias(_r_b_values, _r_b_grads, _r_b_velocities, _r_b_m1, _r_b_m2, _r_b_timesteps, _r_b_decays);
  }
}

bool GRURNNLayer::has_bias() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  return _has_bias_neuron;
}

void GRURNNLayer::calculate_forward_feed(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t batch_size = batch_gradients_and_outputs.size();
  const size_t N_prev = previous_layer.get_number_neurons();
  const size_t N_this = get_number_neurons();

  std::vector<double> sample_inputs = batch_gradients_and_outputs[0].get_rnn_outputs(previous_layer.get_layer_index());
  if (sample_inputs.empty())
  {
      sample_inputs = batch_gradients_and_outputs[0].get_outputs(previous_layer.get_layer_index());
  }
  const size_t num_time_steps = N_prev > 0 ? sample_inputs.size() / N_prev : 0;

  // Flattened storage:
  // batch_output_sequences: [batch_size * num_time_steps * N_this]
  // batch_last_output_sequences: [batch_size * N_this]
  std::vector<double> batch_output_sequences(batch_size * num_time_steps * N_this, 0.0);
  std::vector<double> batch_last_output_sequences(batch_size * N_this, 0.0);

  auto run_forward_pass = [&](size_t start, size_t end)
  {
    // Temporary buffers for accumulation
    // We need 3 separate accumulators for z, r, and h_hat (candidate)
    std::vector<double> z_pre(N_this);
    std::vector<double> r_pre(N_this);
    std::vector<double> h_hat_pre(N_this);
    
    // To store intermediate values for BPTT: z, r, h_hat_pre
    // We will pack them into the HiddenState's pre_activation_sums: [z, r, h_hat_pre]
    std::vector<double> packed_bptt_states(3 * N_this);

    std::vector<double> current_h(N_this, 0.0); // h_t
    std::vector<double> prev_h(N_this, 0.0);    // h_{t-1}

    for (size_t b = start; b < end; ++b)
    {
      // Reset initial hidden state for the sequence
      std::fill(prev_h.begin(), prev_h.end(), 0.0);

      for (size_t t = 0; t < num_time_steps; ++t)
      {
        // 1. Initialize with biases
        if (has_bias())
        {
          for (size_t j = 0; j < N_this; ++j)
          {
            z_pre[j] = _z_b_values[j];
            r_pre[j] = _r_b_values[j];
            h_hat_pre[j] = get_bias_value((unsigned)j); // Candidate bias stored in base class _b_values
          }
        }
        else
        {
          std::fill(z_pre.begin(), z_pre.end(), 0.0);
          std::fill(r_pre.begin(), r_pre.end(), 0.0);
          std::fill(h_hat_pre.begin(), h_hat_pre.end(), 0.0);
        }

        // 2. Input contributions (W * x_t)
        if (get_layer_type() != LayerType::Input)
        {
          const auto& prev_inputs_rnn = batch_gradients_and_outputs[b].get_rnn_outputs(previous_layer.get_layer_index());
          const auto& prev_inputs_std = batch_gradients_and_outputs[b].get_outputs(previous_layer.get_layer_index());
          const bool use_rnn_input = !prev_inputs_rnn.empty();
          const double* prev_inputs_ptr = use_rnn_input ? prev_inputs_rnn.data() : prev_inputs_std.data();
          const size_t prev_inputs_size = use_rnn_input ? prev_inputs_rnn.size() : prev_inputs_std.size();

          constexpr size_t BLOCK_SIZE = 32;
          for (size_t i0 = 0; i0 < N_prev; i0 += BLOCK_SIZE)
          {
             size_t i_limit = std::min(i0 + BLOCK_SIZE, N_prev);
             for (size_t j0 = 0; j0 < N_this; j0 += BLOCK_SIZE)
             {
               size_t j_limit = std::min(j0 + BLOCK_SIZE, N_this);
               
               for (size_t i = i0; i < i_limit; ++i)
               {
                 double val = 0.0;
                 if (prev_inputs_size == N_prev) val = prev_inputs_ptr[i];
                 else if (prev_inputs_size >= (t + 1) * N_prev) val = prev_inputs_ptr[t * N_prev + i];
                 
                 if (val == 0.0) continue;

                 for (size_t j = j0; j < j_limit; ++j)
                 {
                   z_pre[j] += val * _z_w_values[i * N_this + j];
                   r_pre[j] += val * _r_w_values[i * N_this + j];
                   h_hat_pre[j] += val * get_weight_value((unsigned)i, (unsigned)j); // Base weights for candidate
                 }
               }
             }
          }
        }

        // 3. Recurrent contributions (U * h_{t-1}) for Z and R gates
        //    And calculate gates Z and R immediately to use for H_hat
        //    Optimized: We can't fully calculate Z/R until this loop finishes, but we can accumulate U*h
        
        const double* h_prev_ptr = prev_h.data();
        
        // Accumulate U * h_{t-1} for Z and R
        constexpr size_t BLOCK_SIZE = 32;
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
                        z_pre[j] += h_val * _z_rw_values[i * N_this + j];
                        r_pre[j] += h_val * _r_rw_values[i * N_this + j];
                    }
                }
            }
        }

        // 4. Calculate Gate Activations (Sigmoid)
        for (size_t j = 0; j < N_this; ++j)
        {
            double z_val = 1.0 / (1.0 + std::exp(-z_pre[j]));
            double r_val = 1.0 / (1.0 + std::exp(-r_pre[j]));
            
            // Store for BPTT (using packed structure)
            packed_bptt_states[j] = z_val;          // Index 0..N-1
            packed_bptt_states[N_this + j] = r_val; // Index N..2N-1
        }

        // 5. Candidate Recurrent Contribution: U_h * (r_t * h_{t-1})
        for (size_t i0 = 0; i0 < N_this; i0 += BLOCK_SIZE)
        {
            size_t i_limit = std::min(i0 + BLOCK_SIZE, N_this);
            for (size_t j0 = 0; j0 < N_this; j0 += BLOCK_SIZE)
            {
                size_t j_limit = std::min(j0 + BLOCK_SIZE, N_this);
                for (size_t i = i0; i < i_limit; ++i)
                {
                    // r_i * h_{t-1, i}
                    // Note: r is at packed_bptt_states[N_this + i]
                    const double r_val = packed_bptt_states[N_this + i];
                    const double gated_h = r_val * h_prev_ptr[i];
                    
                    if (gated_h == 0.0) continue;
                    for (size_t j = j0; j < j_limit; ++j)
                    {
                        h_hat_pre[j] += gated_h * get_recurrent_weight_value((unsigned)i, (unsigned)j);
                    }
                }
            }
        }

        // 6. Residual Connections
        if (!batch_residual_output_values.empty())
        {
          if (batch_residual_output_values[b].size() == N_this)
          {
            for (size_t j = 0; j < N_this; ++j)
            {
              // Add residual to the candidate state (standard ResNet practice)
              h_hat_pre[j] += batch_residual_output_values[b][j];
            }
          }
        }

        // 7. Final State Update
        const size_t seq_offset = (b * num_time_steps + t) * N_this;
        const size_t last_offset = b * N_this;

        for (size_t j = 0; j < N_this; ++j)
        {
          packed_bptt_states[2 * N_this + j] = h_hat_pre[j]; // Store h_hat pre-activation
          
          double h_hat = get_activation().activate(h_hat_pre[j]);
          
          if (is_training && get_neuron((unsigned)j).is_dropout())
          {
              const auto& neuron = get_neuron((unsigned)j);
              if (neuron.must_randomly_drop()) h_hat = 0.0;
              else h_hat /= (1.0 - neuron.get_dropout_rate());
          }

          double z_val = packed_bptt_states[j];
          double h_val = (1.0 - z_val) * prev_h[j] + z_val * h_hat;
          
          current_h[j] = h_val;
          batch_output_sequences[seq_offset + j] = h_val;
          if (t == num_time_steps - 1) batch_last_output_sequences[last_offset + j] = h_val;
        }

        // Save state
        batch_hidden_states[b].at(get_layer_index())[t].set_pre_activation_sums(packed_bptt_states);
        batch_hidden_states[b].at(get_layer_index())[t].set_hidden_state_values(current_h);
        
        // Update prev_h for next step
        std::copy(current_h.begin(), current_h.end(), prev_h.begin());
      }
    }
  };

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (batch_size < (num_threads * 2))
  {
    run_forward_pass(0, batch_size);
  }
  else
  {
    size_t chunk_size = batch_size / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t start = t * chunk_size;
      size_t end = (t == num_threads - 1) ? batch_size : start + chunk_size;
      _task_queue_pool->enqueue([=]()
        {
          run_forward_pass(start, end);
        });
    }
    _task_queue_pool->get();
  }

  for (size_t b = 0; b < batch_size; ++b)
  {
    auto start_seq = batch_output_sequences.begin() + b * num_time_steps * N_this;
    auto end_seq = start_seq + num_time_steps * N_this;
    std::vector<double> seq_vec(start_seq, end_seq);
    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), seq_vec);
    
    auto start_last = batch_last_output_sequences.begin() + b * N_this;
    auto end_last = start_last + N_this;
    std::vector<double> last_vec(start_last, end_last);
    batch_gradients_and_outputs[b].set_outputs(get_layer_index(), last_vec);
  }
}

void GRURNNLayer::calculate_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs,
  ErrorCalculation::type error_calculation_type) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  switch (error_calculation_type)
  {
  case ErrorCalculation::type::mse:
    return calculate_mse_error_deltas(deltas, target_outputs, given_outputs);
  case ErrorCalculation::type::bce_loss:
    return calculate_bce_error_deltas(deltas, target_outputs, given_outputs);
  default:
    Logger::panic("ErrorCalculation type is not supported for GRURNNLayer!");
  }
}

void GRURNNLayer::calculate_bce_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_total = get_number_neurons();
  const double denom = static_cast<double>(N_total);

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    deltas[neuron_index] = (given_outputs[neuron_index] - target_outputs[neuron_index]) / denom;
  }
}

void GRURNNLayer::calculate_mse_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t N_total = get_number_neurons();
  const double denom = static_cast<double>(N_total);

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    deltas[neuron_index] = (given_outputs[neuron_index] - target_outputs[neuron_index]) / denom;
  }
}

void GRURNNLayer::calculate_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  ErrorCalculation::type error_calculation_type) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t batch_size = batch_gradients_and_outputs.size();
  const size_t N_total = get_number_neurons();

  auto run_output_gradients = [&](size_t start, size_t end)
  {
    std::vector<double> gradients(N_total, 0.0);
    std::vector<double> deltas(N_total, 0.0);

    for (size_t b = start; b < end; b++)
    {
      const auto& given_outputs = batch_gradients_and_outputs[b].get_outputs(get_layer_index());
      const auto& target_outputs = *(target_outputs_begin + b);
      
      std::fill(gradients.begin(), gradients.end(), 0.0);

      if (given_outputs.size() == N_total)
      {
        calculate_error_deltas(deltas, target_outputs, given_outputs, error_calculation_type);
        // Direct assignment of deltas (dL/dh) because GRU output is h_t, not activation(h_t)
        for (unsigned j = 0; j < N_total; ++j) gradients[j] = deltas[j];
      }
      else if (given_outputs.size() >= N_total && N_total > 0)
      {
        const size_t num_time_steps = given_outputs.size() / N_total;
        for (unsigned j = 0; j < N_total; ++j)
        {
          const size_t last_idx = (num_time_steps - 1) * N_total + j;
          const double target = (j < target_outputs.size()) ? target_outputs[j] : 0.0;
          const double delta = (given_outputs[last_idx] - target) / static_cast<double>(N_total);
          gradients[j] = delta;
        }
      }
      batch_gradients_and_outputs[b].set_gradients(get_layer_index(), gradients);
    }
  };

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (batch_size < (num_threads * 2))
  {
    run_output_gradients(0, batch_size);
  }
  else
  {
    size_t chunk_size = batch_size / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t start = t * chunk_size;
      size_t end = (t == num_threads - 1) ? batch_size : start + chunk_size;
      _task_queue_pool->enqueue([=]() {
        run_output_gradients( start, end);
      });
    }
    _task_queue_pool->get();
  }
}

void GRURNNLayer::calculate_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<std::vector<double>>& batch_next_grad_matrix,
  const std::vector<HiddenStates>& batch_hidden_states,
  int bptt_max_ticks) const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  const size_t batch_size = batch_gradients_and_outputs.size();
  const size_t N_this = get_number_neurons();
  const size_t N_next = next_layer.get_number_output_neurons();

  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  if (num_time_steps == 0 || N_this == 0) return;

  const int t_start = static_cast<int>(num_time_steps) - 1;
  int t_end = (bptt_max_ticks > 0) ? std::max(0, t_start - bptt_max_ticks + 1) : 0;

  auto run_hidden_gradients = [&](size_t start, size_t end)
  {
    size_t chunk_count = end - start;
    std::vector<double> chunk_rnn_grad_matrix(chunk_count * num_time_steps * N_this, 0.0); // dL/dx_t (approx)

    // dL/dh_t (gradient w.r.t hidden state)
    // We accumulate this from next layer and next time step.
    std::vector<double> d_next_h(chunk_count * N_this, 0.0);

    // Precompute gradients from next layer (all time steps)
    std::vector<double> chunk_grad_from_next_all_t(chunk_count * num_time_steps * N_this, 0.0);

    for (size_t i = 0; i < N_this; ++i)
    {
      for (size_t k = 0; k < N_next; ++k)
      {
        const double w_ik = next_layer.get_weight_value((unsigned)i, (unsigned)k);
        if (w_ik == 0.0) continue;
        for (size_t b_idx = 0; b_idx < chunk_count; ++b_idx)
        {
          size_t b = start + b_idx;
          const auto& next_grad_matrix = batch_next_grad_matrix[b];
          if (next_grad_matrix.size() == N_next * num_time_steps)
          {
            for (int t = t_start; t >= t_end; --t)
            {
              chunk_grad_from_next_all_t[(b_idx * num_time_steps + t) * N_this + i] += next_grad_matrix[t * N_next + k] * w_ik;
            }
          }
          else if (next_grad_matrix.size() == N_next)
          {
            chunk_grad_from_next_all_t[(b_idx * num_time_steps + t_start) * N_this + i] += next_grad_matrix[k] * w_ik;
          }
        }
      }
    }

    // BPTT Loop
    for (int t = t_start; t >= t_end; --t)
    {
      std::vector<double> chunk_dz(chunk_count * N_this);
      std::vector<double> chunk_dr(chunk_count * N_this);
      std::vector<double> chunk_dh_hat(chunk_count * N_this);
      std::vector<double> chunk_dh_prev_accum(chunk_count * N_this, 0.0); // Next d_next_h

      for (size_t b_idx = 0; b_idx < chunk_count; ++b_idx)
      {
        size_t b = start + b_idx;
        const size_t t_offset = (b_idx * num_time_steps + t) * N_this;
        const auto& packed_states = batch_hidden_states[b].at(get_layer_index())[t].get_pre_activation_sums();
        
        const double* z_vals = &packed_states[0];
        const double* r_vals = &packed_states[N_this];
        const double* h_hat_pre_vals = &packed_states[2 * N_this];
        const double* h_prev_vals = (t > 0) ? batch_hidden_states[b].at(get_layer_index())[t - 1].get_hidden_state_values().data() : nullptr;

        for (size_t j = 0; j < N_this; ++j)
        {
          double dh = chunk_grad_from_next_all_t[t_offset + j] + d_next_h[b_idx * N_this + j];
          
          double z = z_vals[j];
          double h_hat = get_activation().activate(h_hat_pre_vals[j]);
          double h_prev = (h_prev_vals) ? h_prev_vals[j] : 0.0;
          
          double d_z = dh * (h_hat - h_prev);
          double d_z_pre = d_z * z * (1.0 - z);
          
          double d_h_hat = dh * z;
          double d_h_hat_pre = d_h_hat * get_activation().activate_derivative(h_hat_pre_vals[j]);
          
          double d_h_prev_direct = dh * (1.0 - z);

          chunk_dz[b_idx * N_this + j] = d_z_pre;
          chunk_dh_hat[b_idx * N_this + j] = d_h_hat_pre;
          chunk_dh_prev_accum[b_idx * N_this + j] = d_h_prev_direct;
        }
      }

      // Compute d_r_pre
      // dL/dr = (U_h^T * d_h_hat_pre) * h_{t-1}
      std::vector<double> temp_Uh_T_dh_hat(chunk_count * N_this, 0.0);
      
      for (size_t i = 0; i < N_this; ++i) 
      {
          for (size_t j = 0; j < N_this; ++j) 
          {
             double w = get_recurrent_weight_value((unsigned)i, (unsigned)j);
             if (w == 0.0) continue;
             for (size_t b_idx = 0; b_idx < chunk_count; ++b_idx)
             {
                 temp_Uh_T_dh_hat[b_idx * N_this + i] += chunk_dh_hat[b_idx * N_this + j] * w;
             }
          }
      }

      for (size_t b_idx = 0; b_idx < chunk_count; ++b_idx)
      {
          size_t b = start + b_idx;
          const auto& packed_states = batch_hidden_states[b].at(get_layer_index())[t].get_pre_activation_sums();
          const double* r_vals = &packed_states[N_this];
          const double* h_prev_vals = (t > 0) ? batch_hidden_states[b].at(get_layer_index())[t - 1].get_hidden_state_values().data() : nullptr;

          for (size_t i = 0; i < N_this; ++i)
          {
              double grad_rh = temp_Uh_T_dh_hat[b_idx * N_this + i];
              double h_prev = (h_prev_vals) ? h_prev_vals[i] : 0.0;
              double r = r_vals[i];

              double d_r = grad_rh * h_prev;
              double d_r_pre = d_r * r * (1.0 - r);
              chunk_dr[b_idx * N_this + i] = d_r_pre;
              chunk_dh_prev_accum[b_idx * N_this + i] += grad_rh * r;
          }
      }

      // Propagate Back to h_{t-1} (Accumulate to d_next_h)
      // dL/dh_{t-1} += U_z^T * d_z_pre + U_r^T * d_r_pre
      for (size_t i = 0; i < N_this; ++i) 
      {
          for (size_t j = 0; j < N_this; ++j) 
          {
             double w_z = _z_rw_values[i * N_this + j];
             double w_r = _r_rw_values[i * N_this + j];
             
             for (size_t b_idx = 0; b_idx < chunk_count; ++b_idx)
             {
                 double dz = chunk_dz[b_idx * N_this + j];
                 double dr = chunk_dr[b_idx * N_this + j];
                 chunk_dh_prev_accum[b_idx * N_this + i] += dz * w_z + dr * w_r;
             }
          }
      }
      
      // Store accumulators for next time step
      d_next_h = chunk_dh_prev_accum;

      // NOTE: We do not calculate dL/dx (Input gradients) here because 
      // Layer architecture expects dL/dnet to be exposed for Previous Layer.
      // But standard Previous Layer (FFLayer/RNNLayer) expects a single gradient vector.
      // For proper GRU support, NeuralNetwork must handle multiple input matrices.
      // For now, we populate 'rnn_gradients' with dL/dh_hat_pre as a proxy, 
      // effectively treating the Candidate path as the "main" path for backprop to previous layer.
      for (size_t b_idx = 0; b_idx < chunk_count; ++b_idx)
      {
          size_t b = start + b_idx;
          for(size_t j=0; j<N_this; ++j)
          {
              chunk_rnn_grad_matrix[(b_idx * num_time_steps + t) * N_this + j] = chunk_dh_hat[b_idx * N_this + j];
          }
      }
    }

    // Copy results back to batch vectors
    for (size_t b_idx = 0; b_idx < chunk_count; ++b_idx)
    {
      size_t b = start + b_idx;
      auto rnn_grad_start = chunk_rnn_grad_matrix.begin() + b_idx * num_time_steps * N_this;
      std::vector<double> rnn_grad_vec(rnn_grad_start, rnn_grad_start + num_time_steps * N_this);
      batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), rnn_grad_vec);
      // Gradients (non-RNN) not used for BPTT layers usually, but we set them for compatibility
      batch_gradients_and_outputs[b].set_gradients(get_layer_index(), std::vector<double>(N_this, 0.0));
    }
  };

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (batch_size < (num_threads * 2))
  {
    run_hidden_gradients(0, batch_size);
  }
  else
  {
    size_t chunk_size = batch_size / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t start = t * chunk_size;
      size_t end = (t == num_threads - 1) ? batch_size : start + chunk_size;
      _task_queue_pool->enqueue([=]()
        {
          run_hidden_gradients(start, end);
        });
    }
    _task_queue_pool->get();
  }
}

void GRURNNLayer::apply_recurrent_weight_gradient(unsigned from_neuron, unsigned to_neuron, double gradient, double learning_rate, double clipping_scale)
{
    const unsigned idx = from_neuron * get_number_neurons() + to_neuron;
    
    double final_gradient = gradient * clipping_scale;
    if (get_optimiser_type() == OptimiserType::SGD && _rw_decays[idx] > 0.0)
    {
      final_gradient += _rw_decays[idx] * _rw_values[idx];
    }

    switch (get_optimiser_type())
    {
        case OptimiserType::None: {
            _rw_values[idx] -= learning_rate * final_gradient;
            _rw_grads[idx] = final_gradient;
            break;
        }
        case OptimiserType::SGD: {
            _rw_velocities[idx] = get_activation().momentum() * _rw_velocities[idx] + final_gradient;
            _rw_values[idx] -= learning_rate * _rw_velocities[idx];
            _rw_grads[idx] = final_gradient;
            break;
        }
        case OptimiserType::Adam:
        case OptimiserType::AdamW:
        case OptimiserType::Nadam:
        case OptimiserType::NadamW: {
            const double beta1 = 0.9;
            const double beta2 = 0.999;
            const double epsilon = 1e-8;

            _rw_timesteps[idx]++;
            _rw_m1[idx] = beta1 * _rw_m1[idx] + (1.0 - beta1) * final_gradient;
            _rw_m2[idx] = beta2 * _rw_m2[idx] + (1.0 - beta2) * final_gradient * final_gradient;

            double m_hat = _rw_m1[idx] / (1.0 - std::pow(beta1, _rw_timesteps[idx]));
            double v_hat = _rw_m2[idx] / (1.0 - std::pow(beta2, _rw_timesteps[idx]));
            
            double decay = (get_optimiser_type() == OptimiserType::AdamW || get_optimiser_type() == OptimiserType::NadamW) ? (1.0 - learning_rate * _rw_decays[idx]) : 1.0;

            _rw_values[idx] = _rw_values[idx] * decay - learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            _rw_grads[idx] = final_gradient;
            break;
        }
        default:
            break;
    }
}

double GRURNNLayer::get_recurrent_weight_value(unsigned from_neuron, unsigned to_neuron) const
{
    return _rw_values[from_neuron * get_number_neurons() + to_neuron];
}

Layer* GRURNNLayer::clone() const
{
  MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
  return new GRURNNLayer(*this);
}