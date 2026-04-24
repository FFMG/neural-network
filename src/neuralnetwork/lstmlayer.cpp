#include "./libraries/instrumentor.h"
#include "lstmlayer.h"
#include "logger.h"
#include <algorithm>
#include <cmath>

LSTMLayer::LSTMLayer(
  unsigned layer_index,
  unsigned num_neurons_in_previous_layer, 
  unsigned num_neurons_in_this_layer, 
  double weight_decay,
  LayerRole layer_role,
  const activation& activation_method, 
  const OptimiserType& optimiser_type, 
  int residual_layer_number,
  double dropout_rate,
  ResidualProjector* residual_projector,
  int number_of_threads,
  bool has_bias,
  double momentum
) : LSTMLayer(
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
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
}

LSTMLayer::LSTMLayer(
  unsigned layer_index,
  unsigned num_neurons_in_previous_layer,
  unsigned num_neurons_in_this_layer,
  const std::vector<double>& weight_decays,
  LayerRole layer_role,
  const activation& activation_method,
  const OptimiserType& optimiser_type,
  int residual_layer_number,
  double dropout_rate,
  ResidualProjector* residual_projector,
  int number_of_threads,
  bool has_bias,
  double momentum
) : Layer(
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
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  initialize_recurrent_weights(weight_decays.empty() ? 0.0 : weight_decays[0]);
}

LSTMLayer::LSTMLayer(
  unsigned layer_index,
  const LayerRole layer_role,
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
  // Forget Gate (f)
  const std::vector<double>& f_w_values,
  const std::vector<double>& f_w_grads,
  const std::vector<double>& f_w_velocities,
  const std::vector<double>& f_w_m1,
  const std::vector<double>& f_w_m2,
  const std::vector<long long>& f_w_timesteps,
  const std::vector<double>& f_w_decays,
  const std::vector<double>& f_rw_values,
  const std::vector<double>& f_rw_grads,
  const std::vector<double>& f_rw_velocities,
  const std::vector<double>& f_rw_m1,
  const std::vector<double>& f_rw_m2,
  const std::vector<long long>& f_rw_timesteps,
  const std::vector<double>& f_rw_decays,
  const std::vector<double>& f_b_values,
  const std::vector<double>& f_b_grads,
  const std::vector<double>& f_b_velocities,
  const std::vector<double>& f_b_m1,
  const std::vector<double>& f_b_m2,
  const std::vector<long long>& f_b_timesteps,
  const std::vector<double>& f_b_decays,
  // Input Gate (i)
  const std::vector<double>& i_w_values,
  const std::vector<double>& i_w_grads,
  const std::vector<double>& i_w_velocities,
  const std::vector<double>& i_w_m1,
  const std::vector<double>& i_w_m2,
  const std::vector<long long>& i_w_timesteps,
  const std::vector<double>& i_w_decays,
  const std::vector<double>& i_rw_values,
  const std::vector<double>& i_rw_grads,
  const std::vector<double>& i_rw_velocities,
  const std::vector<double>& i_rw_m1,
  const std::vector<double>& i_rw_m2,
  const std::vector<long long>& i_rw_timesteps,
  const std::vector<double>& i_rw_decays,
  const std::vector<double>& i_b_values,
  const std::vector<double>& i_b_grads,
  const std::vector<double>& i_b_velocities,
  const std::vector<double>& i_b_m1,
  const std::vector<double>& i_b_m2,
  const std::vector<long long>& i_b_timesteps,
  const std::vector<double>& i_b_decays,
  // Output Gate (o)
  const std::vector<double>& o_w_values,
  const std::vector<double>& o_w_grads,
  const std::vector<double>& o_w_velocities,
  const std::vector<double>& o_w_m1,
  const std::vector<double>& o_w_m2,
  const std::vector<long long>& o_w_timesteps,
  const std::vector<double>& o_w_decays,
  const std::vector<double>& o_rw_values,
  const std::vector<double>& o_rw_grads,
  const std::vector<double>& o_rw_velocities,
  const std::vector<double>& o_rw_m1,
  const std::vector<double>& o_rw_m2,
  const std::vector<long long>& o_rw_timesteps,
  const std::vector<double>& o_rw_decays,
  const std::vector<double>& o_b_values,
  const std::vector<double>& o_b_grads,
  const std::vector<double>& o_b_velocities,
  const std::vector<double>& o_b_m1,
  const std::vector<double>& o_b_m2,
  const std::vector<long long>& o_b_timesteps,
  const std::vector<double>& o_b_decays,
  const ResidualProjector* residual_projector,
  int number_of_threads,
  const layer_activation_helper& lah,
  double momentum
) noexcept : Layer(
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
  momentum
),
_rw_values(rw_values),
_rw_grads(rw_grads),
_rw_velocities(rw_velocities),
_rw_m1(rw_m1),
_rw_m2(rw_m2),
_rw_timesteps(rw_timesteps),
_rw_decays(rw_decays),
_f_w_values(f_w_values), _f_w_grads(f_w_grads), _f_w_velocities(f_w_velocities), _f_w_m1(f_w_m1), _f_w_m2(f_w_m2), _f_w_decays(f_w_decays), _f_w_timesteps(f_w_timesteps),
_f_rw_values(f_rw_values), _f_rw_grads(f_rw_grads), _f_rw_velocities(f_rw_velocities), _f_rw_m1(f_rw_m1), _f_rw_m2(f_rw_m2), _f_rw_decays(f_rw_decays), _f_rw_timesteps(f_rw_timesteps),
_f_b_values(f_b_values), _f_b_grads(f_b_grads), _f_b_velocities(f_b_velocities), _f_b_m1(f_b_m1), _f_b_m2(f_b_m2), _f_b_decays(f_b_decays), _f_b_timesteps(f_b_timesteps),
_i_w_values(i_w_values), _i_w_grads(i_w_grads), _i_w_velocities(i_w_velocities), _i_w_m1(i_w_m1), _i_w_m2(i_w_m2), _i_w_decays(i_w_decays), _i_w_timesteps(i_w_timesteps),
_i_rw_values(i_rw_values), _i_rw_grads(i_rw_grads), _i_rw_velocities(i_rw_velocities), _i_rw_m1(i_rw_m1), _i_rw_m2(i_rw_m2), _i_rw_decays(i_rw_decays), _i_rw_timesteps(i_rw_timesteps),
_i_b_values(i_b_values), _i_b_grads(i_b_grads), _i_b_velocities(i_b_velocities), _i_b_m1(i_b_m1), _i_b_m2(i_b_m2), _i_b_decays(i_b_decays), _i_b_timesteps(i_b_timesteps),
_o_w_values(o_w_values), _o_w_grads(o_w_grads), _o_w_velocities(o_w_velocities), _o_w_m1(o_w_m1), _o_w_m2(o_w_m2), _o_w_decays(o_w_decays), _o_w_timesteps(o_w_timesteps),
_o_rw_values(o_rw_values), _o_rw_grads(o_rw_grads), _o_rw_velocities(o_rw_velocities), _o_rw_m1(o_rw_m1), _o_rw_m2(o_rw_m2), _o_rw_decays(o_rw_decays), _o_rw_timesteps(o_rw_timesteps),
_o_b_values(o_b_values), _o_b_grads(o_b_grads), _o_b_velocities(o_b_velocities), _o_b_m1(o_b_m1), _o_b_m2(o_b_m2), _o_b_decays(o_b_decays), _o_b_timesteps(o_b_timesteps)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
}

LSTMLayer::LSTMLayer(const LSTMLayer& src) noexcept :
  Layer(src),
  _rw_values(src._rw_values), _rw_grads(src._rw_grads), _rw_velocities(src._rw_velocities), _rw_m1(src._rw_m1), _rw_m2(src._rw_m2), _rw_timesteps(src._rw_timesteps), _rw_decays(src._rw_decays),
  _f_w_values(src._f_w_values), _f_w_grads(src._f_w_grads), _f_w_velocities(src._f_w_velocities), _f_w_m1(src._f_w_m1), _f_w_m2(src._f_w_m2), _f_w_decays(src._f_w_decays), _f_w_timesteps(src._f_w_timesteps),
  _f_rw_values(src._f_rw_values), _f_rw_grads(src._f_rw_grads), _f_rw_velocities(src._f_rw_velocities), _f_rw_m1(src._f_rw_m1), _f_rw_m2(src._f_rw_m2), _f_rw_decays(src._f_rw_decays), _f_rw_timesteps(src._f_rw_timesteps),
  _f_b_values(src._f_b_values), _f_b_grads(src._f_b_grads), _f_b_velocities(src._f_b_velocities), _f_b_m1(src._f_b_m1), _f_b_m2(src._f_b_m2), _f_b_decays(src._f_b_decays), _f_b_timesteps(src._f_b_timesteps),
  _i_w_values(src._i_w_values), _i_w_grads(src._i_w_grads), _i_w_velocities(src._i_w_velocities), _i_w_m1(src._i_w_m1), _i_w_m2(src._i_w_m2), _i_w_decays(src._i_w_decays), _i_w_timesteps(src._i_w_timesteps),
  _i_rw_values(src._i_rw_values), _i_rw_grads(src._i_rw_grads), _i_rw_velocities(src._i_rw_velocities), _i_rw_m1(src._i_rw_m1), _i_rw_m2(src._i_rw_m2), _i_rw_decays(src._i_rw_decays), _i_rw_timesteps(src._i_rw_timesteps),
  _i_b_values(src._i_b_values), _i_b_grads(src._i_b_grads), _i_b_velocities(src._i_b_velocities), _i_b_m1(src._i_b_m1), _i_b_m2(src._i_b_m2), _i_b_decays(src._i_b_decays), _i_b_timesteps(src._i_b_timesteps),
  _o_w_values(src._o_w_values), _o_w_grads(src._o_w_grads), _o_w_velocities(src._o_w_velocities), _o_w_m1(src._o_w_m1), _o_w_m2(src._o_w_m2), _o_w_decays(src._o_w_decays), _o_w_timesteps(src._o_w_timesteps),
  _o_rw_values(src._o_rw_values), _o_rw_grads(src._o_rw_grads), _o_rw_velocities(src._o_rw_velocities), _o_rw_m1(src._o_rw_m1), _o_rw_m2(src._o_rw_m2), _o_rw_decays(src._o_rw_decays), _o_rw_timesteps(src._o_rw_timesteps),
  _o_b_values(src._o_b_values), _o_b_grads(src._o_b_grads), _o_b_velocities(src._o_b_velocities), _o_b_m1(src._o_b_m1), _o_b_m2(src._o_b_m2), _o_b_decays(src._o_b_decays), _o_b_timesteps(src._o_b_timesteps)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
}

LSTMLayer::LSTMLayer(LSTMLayer&& src) noexcept :
  Layer(std::move(src)),
  _rw_values(std::move(src._rw_values)), _rw_grads(std::move(src._rw_grads)), _rw_velocities(std::move(src._rw_velocities)), _rw_m1(std::move(src._rw_m1)), _rw_m2(std::move(src._rw_m2)), _rw_timesteps(std::move(src._rw_timesteps)), _rw_decays(std::move(src._rw_decays)),
  _f_w_values(std::move(src._f_w_values)), _f_w_grads(std::move(src._f_w_grads)), _f_w_velocities(std::move(src._f_w_velocities)), _f_w_m1(std::move(src._f_w_m1)), _f_w_m2(std::move(src._f_w_m2)), _f_w_decays(std::move(src._f_w_decays)), _f_w_timesteps(std::move(src._f_w_timesteps)),
  _f_rw_values(std::move(src._f_rw_values)), _f_rw_grads(std::move(src._f_rw_grads)), _f_rw_velocities(std::move(src._f_rw_velocities)), _f_rw_m1(std::move(src._f_rw_m1)), _f_rw_m2(std::move(src._f_rw_m2)), _f_rw_decays(std::move(src._f_rw_decays)), _f_rw_timesteps(std::move(src._f_rw_timesteps)),
  _f_b_values(std::move(src._f_b_values)), _f_b_grads(std::move(src._f_b_grads)), _f_b_velocities(std::move(src._f_b_velocities)), _f_b_m1(std::move(src._f_b_m1)), _f_b_m2(std::move(src._f_b_m2)), _f_b_decays(std::move(src._f_b_decays)), _f_b_timesteps(std::move(src._f_b_timesteps)),
  _i_w_values(std::move(src._i_w_values)), _i_w_grads(std::move(src._i_w_grads)), _i_w_velocities(std::move(src._i_w_velocities)), _i_w_m1(std::move(src._i_w_m1)), _i_w_m2(std::move(src._i_w_m2)), _i_w_decays(std::move(src._i_w_decays)), _i_w_timesteps(std::move(src._i_w_timesteps)),
  _i_rw_values(std::move(src._i_rw_values)), _i_rw_grads(std::move(src._i_rw_grads)), _i_rw_velocities(std::move(src._i_rw_velocities)), _i_rw_m1(std::move(src._i_rw_m1)), _i_rw_m2(std::move(src._i_rw_m2)), _i_rw_decays(std::move(src._i_rw_decays)), _i_rw_timesteps(std::move(src._i_rw_timesteps)),
  _i_b_values(std::move(src._i_b_values)), _i_b_grads(std::move(src._i_b_grads)), _i_b_velocities(std::move(src._i_b_velocities)), _i_b_m1(std::move(src._i_b_m1)), _i_b_m2(std::move(src._i_b_m2)), _i_b_decays(std::move(src._i_b_decays)), _i_b_timesteps(std::move(src._i_b_timesteps)),
  _o_w_values(std::move(src._o_w_values)), _o_w_grads(std::move(src._o_w_grads)), _o_w_velocities(std::move(src._o_w_velocities)), _o_w_m1(std::move(src._o_w_m1)), _o_w_m2(std::move(src._o_w_m2)), _o_w_decays(std::move(src._o_w_decays)), _o_w_timesteps(std::move(src._o_w_timesteps)),
  _o_rw_values(std::move(src._o_rw_values)), _o_rw_grads(std::move(src._o_rw_grads)), _o_rw_velocities(std::move(src._o_rw_velocities)), _o_rw_m1(std::move(src._o_rw_m1)), _o_rw_m2(std::move(src._o_rw_m2)), _o_rw_decays(std::move(src._o_rw_decays)), _o_rw_timesteps(std::move(src._o_rw_timesteps)),
  _o_b_values(std::move(src._o_b_values)), _o_b_grads(std::move(src._o_b_grads)), _o_b_velocities(std::move(src._o_b_velocities)), _o_b_m1(std::move(src._o_b_m1)), _o_b_m2(std::move(src._o_b_m2)), _o_b_decays(std::move(src._o_b_decays)), _o_b_timesteps(std::move(src._o_b_timesteps))
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
}

LSTMLayer& LSTMLayer::operator=(const LSTMLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  if (this != &src)
  {
    Layer::operator=(src);
    _rw_values = src._rw_values; _rw_grads = src._rw_grads; _rw_velocities = src._rw_velocities; _rw_m1 = src._rw_m1; _rw_m2 = src._rw_m2; _rw_timesteps = src._rw_timesteps; _rw_decays = src._rw_decays;
    _f_w_values = src._f_w_values; _f_w_grads = src._f_w_grads; _f_w_velocities = src._f_w_velocities; _f_w_m1 = src._f_w_m1; _f_w_m2 = src._f_w_m2; _f_w_decays = src._f_w_decays; _f_w_timesteps = src._f_w_timesteps;
    _f_rw_values = src._f_rw_values; _f_rw_grads = src._f_rw_grads; _f_rw_velocities = src._f_rw_velocities; _f_rw_m1 = src._f_rw_m1; _f_rw_m2 = src._f_rw_m2; _f_rw_decays = src._f_rw_decays; _f_rw_timesteps = src._f_rw_timesteps;
    _f_b_values = src._f_b_values; _f_b_grads = src._f_b_grads; _f_b_velocities = src._f_b_velocities; _f_b_m1 = src._f_b_m1; _f_b_m2 = src._f_b_m2; _f_b_decays = src._f_b_decays; _f_b_timesteps = src._f_b_timesteps;
    _i_w_values = src._i_w_values; _i_w_grads = src._i_w_grads; _i_w_velocities = src._i_w_velocities; _i_w_m1 = src._i_w_m1; _i_w_m2 = src._i_w_m2; _i_w_decays = src._i_w_decays; _i_w_timesteps = src._i_w_timesteps;
    _i_rw_values = src._i_rw_values; _i_rw_grads = src._i_rw_grads; _i_rw_velocities = src._i_rw_velocities; _i_rw_m1 = src._i_rw_m1; _i_rw_m2 = src._i_rw_m2; _i_rw_decays = src._i_rw_decays; _i_rw_timesteps = src._i_rw_timesteps;
    _i_b_values = src._i_b_values; _i_b_grads = src._i_b_grads; _i_b_velocities = src._i_b_velocities; _i_b_m1 = src._i_b_m1; _i_b_m2 = src._i_b_m2; _i_b_decays = src._i_b_decays; _i_b_timesteps = src._i_b_timesteps;
    _o_w_values = src._o_w_values; _o_w_grads = src._o_w_grads; _o_w_velocities = src._o_w_velocities; _o_w_m1 = src._o_w_m1; _o_w_m2 = src._o_w_m2; _o_w_decays = src._o_w_decays; _o_w_timesteps = src._o_w_timesteps;
    _o_rw_values = src._o_rw_values; _o_rw_grads = src._o_rw_grads; _o_rw_velocities = src._o_rw_velocities; _o_rw_m1 = src._o_rw_m1; _o_rw_m2 = src._o_rw_m2; _o_rw_decays = src._o_rw_decays; _o_rw_timesteps = src._o_rw_timesteps;
    _o_b_values = src._o_b_values; _o_b_grads = src._o_b_grads; _o_b_velocities = src._o_b_velocities; _o_b_m1 = src._o_b_m1; _o_b_m2 = src._o_b_m2; _o_b_decays = src._o_b_decays; _o_b_timesteps = src._o_b_timesteps;
  }
  return *this;
}

LSTMLayer& LSTMLayer::operator=(LSTMLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  if (this != &src)
  {
    Layer::operator=(std::move(src));
    _rw_values = std::move(src._rw_values); _rw_grads = std::move(src._rw_grads); _rw_velocities = std::move(src._rw_velocities); _rw_m1 = std::move(src._rw_m1); _rw_m2 = std::move(src._rw_m2); _rw_timesteps = std::move(src._rw_timesteps); _rw_decays = std::move(src._rw_decays);
    _f_w_values = std::move(src._f_w_values); _f_w_grads = std::move(src._f_w_grads); _f_w_velocities = std::move(src._f_w_velocities); _f_w_m1 = std::move(src._f_w_m1); _f_w_m2 = std::move(src._f_w_m2); _f_w_decays = std::move(src._f_w_decays); _f_w_timesteps = std::move(src._f_w_timesteps);
    _f_rw_values = std::move(src._f_rw_values); _f_rw_grads = std::move(src._f_rw_grads); _f_rw_velocities = std::move(src._f_rw_velocities); _f_rw_m1 = std::move(src._f_rw_m1); _f_rw_m2 = std::move(src._f_rw_m2); _f_rw_decays = std::move(src._f_rw_decays); _f_rw_timesteps = std::move(src._f_rw_timesteps);
    _f_b_values = std::move(src._f_b_values); _f_b_grads = std::move(src._f_b_grads); _f_b_velocities = std::move(src._f_b_velocities); _f_b_m1 = std::move(src._f_b_m1); _f_b_m2 = std::move(src._f_b_m2); _f_b_decays = std::move(src._f_b_decays); _f_b_timesteps = std::move(src._f_b_timesteps);
    _i_w_values = std::move(src._i_w_values); _i_w_grads = std::move(src._i_w_grads); _i_w_velocities = std::move(src._i_w_velocities); _i_w_m1 = std::move(src._i_w_m1); _i_w_m2 = std::move(src._i_w_m2); _i_w_decays = std::move(src._i_w_decays); _i_w_timesteps = std::move(src._i_w_timesteps);
    _i_rw_values = std::move(src._i_rw_values); _i_rw_grads = std::move(src._i_rw_grads); _i_rw_velocities = std::move(src._i_rw_velocities); _i_rw_m1 = std::move(src._i_rw_m1); _i_rw_m2 = std::move(src._i_rw_m2); _i_rw_decays = std::move(src._i_rw_decays); _i_rw_timesteps = std::move(src._i_rw_timesteps);
    _i_b_values = std::move(src._i_b_values); _i_b_grads = std::move(src._i_b_grads); _i_b_velocities = std::move(src._i_b_velocities); _i_b_m1 = std::move(src._i_b_m1); _i_b_m2 = std::move(src._i_b_m2); _i_b_decays = std::move(src._i_b_decays); _i_b_timesteps = std::move(src._i_b_timesteps);
    _o_w_values = std::move(src._o_w_values); _o_w_grads = std::move(src._o_w_grads); _o_w_velocities = std::move(src._o_w_velocities); _o_w_m1 = std::move(src._o_w_m1); _o_w_m2 = std::move(src._o_w_m2); _o_w_decays = std::move(src._o_w_decays); _o_w_timesteps = std::move(src._o_w_timesteps);
    _o_rw_values = std::move(src._o_rw_values); _o_rw_grads = std::move(src._o_rw_grads); _o_rw_velocities = std::move(src._o_rw_velocities); _o_rw_m1 = std::move(src._o_rw_m1); _o_rw_m2 = std::move(src._o_rw_m2); _o_rw_decays = std::move(src._o_rw_decays); _o_rw_timesteps = std::move(src._o_rw_timesteps);
    _o_b_values = std::move(src._o_b_values); _o_b_grads = std::move(src._o_b_grads); _o_b_velocities = std::move(src._o_b_velocities); _o_b_m1 = std::move(src._o_b_m1); _o_b_m2 = std::move(src._o_b_m2); _o_b_decays = std::move(src._o_b_decays); _o_b_timesteps = std::move(src._o_b_timesteps);
  }
  return *this;
}

LSTMLayer::~LSTMLayer() {}

void LSTMLayer::initialize_recurrent_weights(double weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const auto num_neurons = get_number_neurons();
  const auto num_inputs = get_number_input_neurons();

  const size_t num_rec_weights = static_cast<size_t>(num_neurons) * num_neurons;
  const size_t num_inp_weights = static_cast<size_t>(num_inputs) * num_neurons;

  init_weights(_rw_values, _rw_grads, _rw_velocities, _rw_m1, _rw_m2, _rw_timesteps, _rw_decays, num_rec_weights, false, weight_decay);
  if (num_inputs > 0) init_weights(_f_w_values, _f_w_grads, _f_w_velocities, _f_w_m1, _f_w_m2, _f_w_timesteps, _f_w_decays, num_inp_weights, true, weight_decay);
  init_weights(_f_rw_values, _f_rw_grads, _f_rw_velocities, _f_rw_m1, _f_rw_m2, _f_rw_timesteps, _f_rw_decays, num_rec_weights, false, weight_decay);
  if (has_bias()) init_bias(_f_b_values, _f_b_grads, _f_b_velocities, _f_b_m1, _f_b_m2, _f_b_timesteps, _f_b_decays);
  if (num_inputs > 0) init_weights(_i_w_values, _i_w_grads, _i_w_velocities, _i_w_m1, _i_w_m2, _i_w_timesteps, _i_w_decays, num_inp_weights, true, weight_decay);
  init_weights(_i_rw_values, _i_rw_grads, _i_rw_velocities, _i_rw_m1, _i_rw_m2, _i_rw_timesteps, _i_rw_decays, num_rec_weights, false, weight_decay);
  if (has_bias()) init_bias(_i_b_values, _i_b_grads, _i_b_velocities, _i_b_m1, _i_b_m2, _i_b_timesteps, _i_b_decays);
  if (num_inputs > 0) init_weights(_o_w_values, _o_w_grads, _o_w_velocities, _o_w_m1, _o_w_m2, _o_w_timesteps, _o_w_decays, num_inp_weights, true, weight_decay);
  init_weights(_o_rw_values, _o_rw_grads, _o_rw_velocities, _o_rw_m1, _o_rw_m2, _o_rw_timesteps, _o_rw_decays, num_rec_weights, false, weight_decay);
  if (has_bias()) init_bias(_o_b_values, _o_b_grads, _o_b_velocities, _o_b_m1, _o_b_m2, _o_b_timesteps, _o_b_decays);
}

void LSTMLayer::init_weights(std::vector<double>& values, std::vector<double>& grads, std::vector<double>& velocities, std::vector<double>& m1, std::vector<double>& m2, std::vector<long long>& timesteps, std::vector<double>& decays, size_t size, bool is_input, double weight_decay) const
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const auto num_neurons = get_number_output_neurons();
  const auto num_inputs = get_number_input_neurons();
  values.resize(size);
  const unsigned f_in = is_input ? num_inputs : num_neurons;
  for (size_t i = 0; i < size; ++i) values[i] = get_activation().weight_initialization(f_in, num_neurons);
  grads.assign(size, 0.0);
  velocities.assign(size, 0.0);
  m1.assign(size, 0.0);
  m2.assign(size, 0.0);
  timesteps.assign(size, 0);
  decays.assign(size, weight_decay);
}

void LSTMLayer::init_bias(std::vector<double>& values, std::vector<double>& grads, std::vector<double>& velocities, std::vector<double>& m1, std::vector<double>& m2, std::vector<long long>& timesteps, std::vector<double>& decays) const
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const auto num_neurons = get_number_neurons();
  values.assign(num_neurons, 0.0);
  grads.assign(num_neurons, 0.0);
  velocities.assign(num_neurons, 0.0);
  m1.assign(num_neurons, 0.0);
  m2.assign(num_neurons, 0.0);
  timesteps.assign(num_neurons, 0);
  decays.assign(num_neurons, 0.0);
}

void LSTMLayer::calculate_forward_feed(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  if (batch_size == 0) return;
  const size_t N_prev = previous_layer.get_number_neurons();
  const size_t N_this = get_number_neurons();
  std::vector<double> flattened_batch_inputs;
  size_t num_time_steps = 0;
  const unsigned prev_layer_index = previous_layer.get_layer_index();
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    if (!rnn_in.empty())
    {
      const size_t t = rnn_in.size() / N_prev;
      if (num_time_steps == 0) { num_time_steps = t; flattened_batch_inputs.resize(batch_size * num_time_steps * N_prev); }
      std::copy(rnn_in.begin(), rnn_in.end(), flattened_batch_inputs.begin() + b * num_time_steps * N_prev);
    }
    else
    {
      const auto std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
      if (std_in.size() == N_prev)
      {
        if (num_time_steps == 0) { num_time_steps = 1; flattened_batch_inputs.resize(batch_size * num_time_steps * N_prev); }
        for (size_t t = 0; t < num_time_steps; ++t) std::copy(std_in.begin(), std_in.end(), flattened_batch_inputs.begin() + (b * num_time_steps + t) * N_prev);
      }
    }
  }
  if (num_time_steps == 0) return;
  std::vector<double> batch_output_sequences(batch_size * num_time_steps * N_this, 0.0);
  auto run_forward_pass = [&](size_t start, size_t end)
  {
    std::vector<double> f_pre(N_this), i_pre(N_this), o_pre(N_this), g_pre(N_this);
    std::vector<double> packed_bptt_states(4 * N_this);
    for (size_t b = start; b < end; ++b)
    {
      std::vector<double> prev_h(N_this, 0.0);
      std::vector<double> prev_c(N_this, 0.0);
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        if (has_bias())
        {
          std::copy(_f_b_values.begin(), _f_b_values.end(), f_pre.begin());
          std::copy(_i_b_values.begin(), _i_b_values.end(), i_pre.begin());
          std::copy(_o_b_values.begin(), _o_b_values.end(), o_pre.begin());
          std::copy(get_b_values().begin(), get_b_values().end(), g_pre.begin());
        }
        else
        {
          std::fill(f_pre.begin(), f_pre.end(), 0.0);
          std::fill(i_pre.begin(), i_pre.end(), 0.0);
          std::fill(o_pre.begin(), o_pre.end(), 0.0);
          std::fill(g_pre.begin(), g_pre.end(), 0.0);
        }
        const double* x_t = &flattened_batch_inputs[(b * num_time_steps + t) * N_prev];
        for (size_t i = 0; i < N_prev; ++i)
        {
          const double x_val = x_t[i];
          if (x_val == 0.0) continue;
          for (size_t j = 0; j < N_this; ++j)
          {
            f_pre[j] += x_val * _f_w_values[i * N_this + j];
            i_pre[j] += x_val * _i_w_values[i * N_this + j];
            o_pre[j] += x_val * _o_w_values[i * N_this + j];
            g_pre[j] += x_val * get_w_values()[i * N_this + j];
          }
        }
        for (size_t i = 0; i < N_this; ++i)
        {
          const double h_val = prev_h[i];
          if (h_val == 0.0) continue;
          for (size_t j = 0; j < N_this; ++j)
          {
            f_pre[j] += h_val * _f_rw_values[i * N_this + j];
            i_pre[j] += h_val * _i_rw_values[i * N_this + j];
            o_pre[j] += h_val * _o_rw_values[i * N_this + j];
            g_pre[j] += h_val * _rw_values[i * N_this + j];
          }
        }
        if (!batch_residual_output_values.empty() && batch_residual_output_values[b].size() == N_this)
        {
          for (size_t j = 0; j < N_this; ++j) g_pre[j] += batch_residual_output_values[b][j];
        }
        std::vector<double> current_c(N_this);
        std::vector<double> current_h(N_this);
        for (size_t j = 0; j < N_this; ++j)
        {
          double f = 1.0 / (1.0 + std::exp(-f_pre[j]));
          double i = 1.0 / (1.0 + std::exp(-i_pre[j]));
          double o = 1.0 / (1.0 + std::exp(-o_pre[j]));
          double g = std::tanh(g_pre[j]);
          packed_bptt_states[j] = f;
          packed_bptt_states[N_this + j] = i;
          packed_bptt_states[2 * N_this + j] = o;
          packed_bptt_states[3 * N_this + j] = g_pre[j];
          current_c[j] = f * prev_c[j] + i * g;
          current_h[j] = o * std::tanh(current_c[j]);
        }
        batch_hidden_states[b].at(get_layer_index())[t].set_pre_activation_sums(packed_bptt_states);
        batch_hidden_states[b].at(get_layer_index())[t].set_cell_state_values(current_c);
        for (size_t j = 0; j < N_this; ++j)
        {
          double out = current_h[j];
          if (is_training && get_neuron((unsigned)j).is_dropout())
          {
            const auto& neuron = get_neuron((unsigned)j);
            if (neuron.must_randomly_drop()) out = 0.0;
            else out /= (1.0 - neuron.get_dropout_rate());
          }
          current_h[j] = out;
          batch_output_sequences[(b * num_time_steps + t) * N_this + j] = out;
        }
        batch_hidden_states[b].at(get_layer_index())[t].set_hidden_state_values(current_h);
        prev_h = current_h;
        prev_c = current_c;
      }
    }
  };
  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads <= 1) run_forward_pass(0, batch_size);
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t size = (batch_size / num_threads) + (t < (batch_size % num_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end) _task_queue_pool->enqueue([run_forward_pass, start, end]() { run_forward_pass(start, end); });
      start = end;
    }
    _task_queue_pool->get();
  }
  for (size_t b = 0; b < batch_size; ++b)
  {
    const double* seq_ptr = &batch_output_sequences[b * num_time_steps * N_this];
    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), std::vector<double>(seq_ptr, seq_ptr + num_time_steps * N_this));

    const double* last_ptr = &batch_output_sequences[(b * num_time_steps + num_time_steps - 1) * N_this];
    batch_gradients_and_outputs[b].set_outputs(get_layer_index(), std::vector<double>(last_ptr, last_ptr + N_this));
  }
}

void LSTMLayer::calculate_output_gradients(std::vector<GradientsAndOutputs>& batch_gradients_and_outputs, std::vector<std::vector<double>>::const_iterator target_outputs_begin, const std::vector<HiddenStates>& batch_hidden_states, size_t batch_size) const
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& states = batch_hidden_states[b].at(get_layer_index());
    const size_t T = states.size();
    std::vector<double> deltas(T * N_this);
    const std::vector<double>& targets = *(target_outputs_begin + b);
    for (size_t t = 0; t < T; ++t)
    {
      const auto& given = states[t].get_hidden_state_values();
      for (size_t j = 0; j < N_this; ++j)
      {
        size_t idx = t * N_this + j;
        if (idx < targets.size()) deltas[idx] = given[j] - targets[idx];
        else deltas[idx] = 0.0;
      }
    }
    // Set BOTH RNN gradients and standard gradients (last step)
    batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), deltas);
    
    std::vector<double> last_step_deltas(N_this);
    std::copy(deltas.end() - N_this, deltas.end(), last_step_deltas.begin());
    batch_gradients_and_outputs[b].set_gradients(get_layer_index(), last_step_deltas);
  }
}

void LSTMLayer::calculate_hidden_gradients(std::vector<GradientsAndOutputs>& batch_gradients_and_outputs, const Layer& next_layer, const std::vector<std::vector<double>>& batch_next_grad_matrix, const std::vector<HiddenStates>& batch_hidden_states, size_t batch_size, int bptt_max_ticks) const
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_next = next_layer.get_number_neurons();
  for (size_t b = 0; b < batch_size; ++b)
  {
    const size_t T = batch_hidden_states[b].at(get_layer_index()).size();
    std::vector<double> hidden_grads(T * N_this, 0.0);
    const double* next_grads = batch_next_grad_matrix[b].data();
    
    // If next_grads is a single step (size N_next), it applies only to the LAST step of this layer
    const bool next_is_seq = (batch_next_grad_matrix[b].size() == T * N_next);

    for (size_t t = 0; t < T; ++t)
    {
      if (!next_is_seq && t < T - 1) continue;
      
      const double* g_next_t = next_is_seq ? &next_grads[t * N_next] : next_grads;
      for (size_t j = 0; j < N_this; ++j)
      {
        double sum = 0.0;
        for (size_t k = 0; k < N_next; ++k) sum += g_next_t[k] * next_layer.get_weight_value((unsigned)j, (unsigned)k);
        hidden_grads[t * N_this + j] = sum;
      }
    }
    // Set BOTH RNN gradients and standard gradients (last step or sum)
    batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), hidden_grads);
    
    std::vector<double> last_step_grads(N_this);
    std::copy(hidden_grads.end() - N_this, hidden_grads.end(), last_step_grads.begin());
    batch_gradients_and_outputs[b].set_gradients(get_layer_index(), last_step_grads);
  }
}

void LSTMLayer::calculate_and_store_gradients(const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs, const std::vector<HiddenStates>& hidden_states, const Layer& previous_layer, size_t batch_size, int bptt_max_ticks)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = previous_layer.get_number_neurons();
  const size_t T = hidden_states[0].at(get_layer_index()).size();
  
  const int t_start = static_cast<int>(T) - 1;
  const int t_end = (bptt_max_ticks > 0) ? std::max(0, t_start - bptt_max_ticks + 1) : 0;

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& layer_states = hidden_states[b].at(get_layer_index());
    const auto& upstream_grads = batch_gradients_and_outputs[b].get_rnn_gradients(get_layer_index());
    if (upstream_grads.empty()) continue;

    std::vector<double> dh_next(N_this, 0.0);
    std::vector<double> dc_next(N_this, 0.0);
    for (int t = t_start; t >= t_end; --t)
    {
      const auto& state = layer_states[t];
      const auto packed = state.get_pre_activation_sums();
      const auto c_curr = state.get_cell_state_values();
      
      const bool has_prev = (t > 0);
      const auto c_prev = has_prev ? layer_states[t - 1].get_cell_state_values() : std::span<double>();
      const auto h_prev = has_prev ? layer_states[t - 1].get_hidden_state_values() : std::span<double>();
      
      const double* f = &packed[0];
      const double* i = &packed[N_this];
      const double* o = &packed[2 * N_this];
      const double* g_pre = &packed[3 * N_this];
      
      std::vector<double> dh_prev(N_this, 0.0);
      for (size_t j = 0; j < N_this; ++j)
      {
        double dh = upstream_grads[t * N_this + j] + dh_next[j];
        double tanh_c = std::tanh(c_curr[j]);
        double do_gate = dh * tanh_c * o[j] * (1.0 - o[j]);
        double dc = dh * o[j] * (1.0 - tanh_c * tanh_c) + dc_next[j];
        double g = std::tanh(g_pre[j]);
        double df = dc * (has_prev ? c_prev[j] : 0.0) * f[j] * (1.0 - f[j]);
        double di = dc * g * i[j] * (1.0 - i[j]);
        double dg = dc * i[j] * (1.0 - g * g);
        _f_b_grads[j] += df; _i_b_grads[j] += di; _o_b_grads[j] += do_gate; _b_grads[j] += dg;
        
        const auto& prev_rnn_out = batch_gradients_and_outputs[b].get_rnn_outputs(previous_layer.get_layer_index());
        const auto& prev_std_out = batch_gradients_and_outputs[b].get_outputs(previous_layer.get_layer_index());
        const double* x_t_base = !prev_rnn_out.empty() ? prev_rnn_out.data() : prev_std_out.data();
        const size_t x_size = !prev_rnn_out.empty() ? prev_rnn_out.size() : prev_std_out.size();

        const double* x_curr = (x_size == T * N_prev) ? &x_t_base[t * N_prev] : x_t_base;
        
        for (size_t k = 0; k < N_prev; ++k)
        {
          _f_w_grads[k * N_this + j] += df * x_curr[k];
          _i_w_grads[k * N_this + j] += di * x_curr[k];
          _o_w_grads[k * N_this + j] += do_gate * x_curr[k];
          _w_grads[k * N_this + j] += dg * x_curr[k];
        }
        for (size_t k = 0; k < N_this; ++k)
        {
          _f_rw_grads[k * N_this + j] += df * h_prev[k];
          _i_rw_grads[k * N_this + j] += di * h_prev[k];
          _o_rw_grads[k * N_this + j] += do_gate * h_prev[k];
          _rw_grads[k * N_this + j] += dg * h_prev[k];
          dh_prev[k] += df * _f_rw_values[k * N_this + j] + di * _i_rw_values[k * N_this + j] + do_gate * _o_rw_values[k * N_this + j] + dg * _rw_values[k * N_this + j];
        }
        dc_next[j] = dc * f[j];
      }
      dh_next = std::move(dh_prev);
    }
  }

  // Normalization
  const int active_ticks = t_start - t_end + 1;
  const double inv_batch_active = 1.0 / (static_cast<double>(batch_size) * active_ticks);
  const double inv_batch_rec = 1.0 / (static_cast<double>(batch_size) * (active_ticks > 1 ? active_ticks - 1 : 1.0));

  auto norm = [](std::vector<double>& v, double factor) { for (double& x : v) x *= factor; };
  norm(_f_w_grads, inv_batch_active); norm(_i_w_grads, inv_batch_active); norm(_o_w_grads, inv_batch_active); norm(_w_grads, inv_batch_active);
  norm(_f_rw_grads, inv_batch_rec); norm(_i_rw_grads, inv_batch_rec); norm(_o_rw_grads, inv_batch_rec); norm(_rw_grads, inv_batch_rec);
  norm(_f_b_grads, inv_batch_active); norm(_i_b_grads, inv_batch_active); norm(_o_b_grads, inv_batch_active); norm(_b_grads, inv_batch_active);
}

double LSTMLayer::get_gradient_norm_sq() const
{
  auto ssq = [](const std::vector<double>& v) { double s = 0; for (double x : v) s += x * x; return s; };
  return ssq(_w_grads) + ssq(_b_grads) + ssq(_rw_grads) + ssq(_f_w_grads) + ssq(_f_b_grads) + ssq(_f_rw_grads) + ssq(_i_w_grads) + ssq(_i_b_grads) + ssq(_i_rw_grads) + ssq(_o_w_grads) + ssq(_o_b_grads) + ssq(_o_rw_grads);
}

void LSTMLayer::zero_gradients()
{
  std::fill(_w_grads.begin(), _w_grads.end(), 0.0); std::fill(_b_grads.begin(), _b_grads.end(), 0.0); std::fill(_rw_grads.begin(), _rw_grads.end(), 0.0);
  std::fill(_f_w_grads.begin(), _f_w_grads.end(), 0.0); std::fill(_f_b_grads.begin(), _f_b_grads.end(), 0.0); std::fill(_f_rw_grads.begin(), _f_rw_grads.end(), 0.0);
  std::fill(_i_w_grads.begin(), _i_w_grads.end(), 0.0); std::fill(_i_b_grads.begin(), _i_b_grads.end(), 0.0); std::fill(_i_rw_grads.begin(), _i_rw_grads.end(), 0.0);
  std::fill(_o_w_grads.begin(), _o_w_grads.end(), 0.0); std::fill(_o_b_grads.begin(), _o_b_grads.end(), 0.0); std::fill(_o_rw_grads.begin(), _o_rw_grads.end(), 0.0);
}

void LSTMLayer::apply_stored_gradients(double learning_rate, double clipping_scale)
{
  auto app = [&](std::vector<double>& v, std::vector<double>& g, std::vector<double>& vel, std::vector<double>& m1, std::vector<double>& m2, std::vector<long long>& ts, const std::vector<double>& dec, bool is_bias) {
    apply_update_to_vector(v, g, vel, m1, m2, ts, dec, learning_rate, clipping_scale, is_bias, get_optimiser_type());
  };
  app(_w_values, _w_grads, _w_velocities, _w_m1, _w_m2, _w_timesteps, _w_decays, false);
  app(_b_values, _b_grads, _b_velocities, _b_m1, _b_m2, _b_timesteps, _b_decays, true);
  app(_rw_values, _rw_grads, _rw_velocities, _rw_m1, _rw_m2, _rw_timesteps, _rw_decays, false);
  app(_f_w_values, _f_w_grads, _f_w_velocities, _f_w_m1, _f_w_m2, _f_w_timesteps, _f_w_decays, false);
  app(_f_b_values, _f_b_grads, _f_b_velocities, _f_b_m1, _f_b_m2, _f_b_timesteps, _f_b_decays, true);
  app(_f_rw_values, _f_rw_grads, _f_rw_velocities, _f_rw_m1, _f_rw_m2, _f_rw_timesteps, _f_rw_decays, false);
  app(_i_w_values, _i_w_grads, _i_w_velocities, _i_w_m1, _i_w_m2, _i_w_timesteps, _i_w_decays, false);
  app(_i_b_values, _i_b_grads, _i_b_velocities, _i_b_m1, _i_b_m2, _i_b_timesteps, _i_b_decays, true);
  app(_i_rw_values, _i_rw_grads, _i_rw_velocities, _i_rw_m1, _i_rw_m2, _i_rw_timesteps, _i_rw_decays, false);
  app(_o_w_values, _o_w_grads, _o_w_velocities, _o_w_m1, _o_w_m2, _o_w_timesteps, _o_w_decays, false);
  app(_o_b_values, _o_b_grads, _o_b_velocities, _o_b_m1, _o_b_m2, _o_b_timesteps, _o_b_decays, true);
  app(_o_rw_values, _o_rw_grads, _o_rw_velocities, _o_rw_m1, _o_rw_m2, _o_rw_timesteps, _o_rw_decays, false);
}

double LSTMLayer::get_recurrent_weight_value(unsigned f, unsigned t) const { return _rw_values[f * get_number_neurons() + t]; }
Layer* LSTMLayer::clone() const { return new LSTMLayer(*this); }
