#include "./libraries/instrumentor.h"
#include "lstmlayer.h"
#include "fflayer.h"
#include "simd_utils.h"
#include "logger.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h>

LSTMLayer::LSTMLayer(unsigned layer_index,
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
    double momentum) : LSTMLayer(
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
  allocate_workspace();
}

LSTMLayer::LSTMLayer(unsigned layer_index,
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
    double momentum) : Layer(
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
  allocate_workspace();
}

LSTMLayer::LSTMLayer(
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
_rw_values(rw_values), _rw_grads(rw_grads), _rw_velocities(rw_velocities), _rw_m1(rw_m1), _rw_m2(rw_m2), _rw_timesteps(rw_timesteps), _rw_decays(rw_decays),
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
  allocate_workspace();
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
  _o_b_values(std::move(src._o_b_values)), _o_b_grads(std::move(src._o_b_grads)), _o_b_velocities(std::move(src._o_b_velocities)), _o_b_m1(std::move(src._o_b_m1)), _o_b_m2(std::move(src._o_b_m2)), _o_b_decays(std::move(src._o_b_decays)), _o_b_timesteps(std::move(src._o_b_timesteps)),
  _thread_workspaces(std::move(src._thread_workspaces))
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
    allocate_workspace();
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
    _thread_workspaces = std::move(src._thread_workspaces);
  }
  return *this;
}

LSTMLayer::~LSTMLayer() 
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
}

Layer* LSTMLayer::clone() const 
{ 
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  return new LSTMLayer(*this); 
}

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
  const unsigned prev_layer_index = previous_layer.get_layer_index();

  // 1. Determine sequence length and flatten inputs
  size_t num_time_steps = 0;
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    if (!rnn_in.empty()) { num_time_steps = rnn_in.size() / N_prev; break; }
    const auto std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
    if (std_in.size() == N_prev) { num_time_steps = 1; break; }
  }
  if (num_time_steps == 0) return;

  std::vector<double> flattened_inputs(batch_size * num_time_steps * N_prev);
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    if (!rnn_in.empty()) std::copy(rnn_in.begin(), rnn_in.end(), flattened_inputs.begin() + b * num_time_steps * N_prev);
    else
    {
      const auto std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
      for (size_t t = 0; t < num_time_steps; ++t) std::copy(std_in.begin(), std_in.end(), flattened_inputs.begin() + (b * num_time_steps + t) * N_prev);
    }
  }

  // 2. Pre-calculate Input-to-Gates (all 4 gates) for all ticks
  // Pre-activations buffer: [Batch x Ticks x 4 x N_this]
  std::vector<double> batch_pre_act(batch_size * num_time_steps * GateCount * N_this, 0.0);

  auto precalc_gates = [&](size_t b_start, size_t b_end)
  {
    for (size_t b = b_start; b < b_end; ++b)
    {
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        const double* x_t = &flattened_inputs[(b * num_time_steps + t) * N_prev];
        double* pre_t = &batch_pre_act[(b * num_time_steps + t) * GateCount * N_this];
        double* f_pre = pre_t;
        double* i_pre = pre_t + N_this;
        double* o_pre = pre_t + 2 * N_this;
        double* g_pre = pre_t + 3 * N_this;

        // Bias initialization
        if (has_bias())
        {
          std::copy(_f_b_values.begin(), _f_b_values.end(), f_pre);
          std::copy(_i_b_values.begin(), _i_b_values.end(), i_pre);
          std::copy(_o_b_values.begin(), _o_b_values.end(), o_pre);
          std::copy(_b_values.begin(), _b_values.end(), g_pre);
        }

        // Input-to-Gates GEMV (vectorized)
        for (size_t i = 0; i < N_prev; ++i)
        {
          const double xi = x_t[i];
          if (xi == 0.0) continue;

          simd::mul_add(xi, &_f_w_values[i * N_this], f_pre, N_this);
          simd::mul_add(xi, &_i_w_values[i * N_this], i_pre, N_this);
          simd::mul_add(xi, &_o_w_values[i * N_this], o_pre, N_this);
          simd::mul_add(xi, &get_w_values()[i * N_this], g_pre, N_this);
        }
      }
    }
  };

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads <= 1) precalc_gates(0, batch_size);
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t size = (batch_size / num_threads) + (t < (batch_size % num_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end) _task_queue_pool->enqueue([&precalc_gates, start, end]() { precalc_gates(start, end); });
      start = end;
    }
    _task_queue_pool->get();
  }

        // Sequential Recurrent Pass and Activations
  std::vector<double> batch_output_sequences(batch_size * num_time_steps * N_this);

  auto recurrent_pass = [&](size_t b_start, size_t b_end)
  {
    std::vector<double> current_h(N_this, 0.0);
    std::vector<double> current_c(N_this, 0.0);
    std::vector<double> packed_bptt(Multiplier * N_this);

    for (size_t b = b_start; b < b_end; ++b)
    {
      std::fill(current_h.begin(), current_h.end(), 0.0);
      std::fill(current_c.begin(), current_c.end(), 0.0);

      for (size_t t = 0; t < num_time_steps; ++t)
      {
        double* pre_t = &batch_pre_act[(b * num_time_steps + t) * GateCount * N_this];
        double* f_pre = pre_t;
        double* i_pre = pre_t + N_this;
        double* o_pre = pre_t + 2 * N_this;
        double* g_pre = pre_t + 3 * N_this;

        // Recurrent-to-Gates
        for (size_t i = 0; i < N_this; ++i)
        {
          const double hi = current_h[i];
          if (hi == 0.0) continue;
          for (size_t j = 0; j < N_this; ++j)
          {
            f_pre[j] += hi * _f_rw_values[i * N_this + j];
            i_pre[j] += hi * _i_rw_values[i * N_this + j];
            o_pre[j] += hi * _o_rw_values[i * N_this + j];
            g_pre[j] += hi * _rw_values[i * N_this + j];
          }
        }

        // Residuals (on candidate gate g)
        if (!batch_residual_output_values.empty() && batch_residual_output_values[b].size() == N_this)
        {
          for (size_t j = 0; j < N_this; ++j) g_pre[j] += batch_residual_output_values[b][j];
        }

        // Activations
        for (size_t j = 0; j < N_this; ++j)
        {
          double f = 1.0 / (1.0 + std::exp(-f_pre[j]));
          double i = 1.0 / (1.0 + std::exp(-i_pre[j]));
          double o = 1.0 / (1.0 + std::exp(-o_pre[j]));
          double g = std::tanh(g_pre[j]);

          packed_bptt[j] = f;
          packed_bptt[N_this + j] = i;
          packed_bptt[2 * N_this + j] = o;
          packed_bptt[3 * N_this + j] = g_pre[j]; // Store pre-tanh for backprop

          current_c[j] = f * current_c[j] + i * g;
          double tanh_c = std::tanh(current_c[j]);
          double out = o * tanh_c;

          // Dropout
          double mask = 1.0;
          if (is_training && get_neuron((unsigned)j).is_dropout())
          {
            const auto& neuron = get_neuron((unsigned)j);
            const double dropout_rate = neuron.get_dropout_rate();
            if (neuron.must_randomly_drop())
            {
              out = 0.0;
              mask = 0.0;
            }
            else
            {
              mask = 1.0 / (1.0 - dropout_rate);
              out *= mask;
            }
          }
          packed_bptt[GateCount * N_this + j] = mask;

          current_h[j] = out;
          batch_output_sequences[(b * num_time_steps + t) * N_this + j] = out;
        }

        // Store states
        auto& state = batch_hidden_states[b].at(get_layer_index())[t];
        state.set_pre_activation_sums(packed_bptt);
        state.set_cell_state_values(current_c);
        state.set_hidden_state_values(current_h);
      }
    }
  };

  if (num_threads <= 1) recurrent_pass(0, batch_size);
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t size = (batch_size / num_threads) + (t < (batch_size % num_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end) _task_queue_pool->enqueue([&recurrent_pass, start, end]() { recurrent_pass(start, end); });
      start = end;
    }
    _task_queue_pool->get();
  }

  // 4. Output GradientsAndOutputs
  for (size_t b = 0; b < batch_size; ++b)
  {
    const double* seq_ptr = &batch_output_sequences[b * num_time_steps * N_this];
    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), std::vector<double>(seq_ptr, seq_ptr + num_time_steps * N_this));
    const double* last_ptr = seq_ptr + (num_time_steps - 1) * N_this;
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
    batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), deltas);
    std::vector<double> last_step_deltas(N_this);
    std::copy(deltas.end() - N_this, deltas.end(), last_step_deltas.begin());
    batch_gradients_and_outputs[b].set_gradients(get_layer_index(), last_step_deltas);
  }
}

void LSTMLayer::calculate_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<std::vector<double>>& batch_next_grad_matrix,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int bptt_max_ticks) const
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  if (batch_size == 0) return;
  const size_t N_this = get_number_neurons();
  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  if (num_time_steps == 0 || N_this == 0) return;
  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads <= 1)
  {
    auto& workspace = get_workspace(0);
    calculate_bptt_batch_chunk(0, batch_size, batch_gradients_and_outputs, next_layer, batch_next_grad_matrix, batch_hidden_states, bptt_max_ticks, workspace, _rw_values_T, _f_rw_values_T, _i_rw_values_T, _o_rw_values_T);
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
            calculate_bptt_batch_chunk(start, end, batch_gradients_and_outputs, next_layer, batch_next_grad_matrix, batch_hidden_states, bptt_max_ticks, workspace, _rw_values_T, _f_rw_values_T, _i_rw_values_T, _o_rw_values_T);
          });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}

void LSTMLayer::calculate_hidden_gradients_from_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<std::vector<double>>& batch_output_gradients,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int bptt_max_ticks) const
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const auto N_this = get_number_neurons();
  if (N_this == 0 || batch_size == 0) return;
  FFLayer proxy(0, N_this, N_this, 0.0, Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, false, 0.0);
  std::vector<double> id(static_cast<size_t>(N_this) * N_this, 0.0);
  for (unsigned i = 0; i < N_this; ++i) id[i * N_this + i] = 1.0;
  proxy.set_w_values(id);
  calculate_hidden_gradients(batch_gradients_and_outputs, proxy, batch_output_gradients, batch_hidden_states, batch_size, bptt_max_ticks);
}

void LSTMLayer::calculate_and_store_gradients(
const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs, const std::vector<HiddenStates>& hidden_states, const Layer& previous_layer, size_t batch_size, int /*bptt_max_ticks*/)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  if (batch_size == 0)
  {
    return;
  }
  const size_t N_this = get_number_neurons();
  const size_t N_prev = previous_layer.get_number_neurons();
  const size_t T = hidden_states[0].at(get_layer_index()).size();
  const unsigned prev_layer_index = previous_layer.get_layer_index();

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  std::vector<std::vector<double>> thread_w_grads(num_threads, std::vector<double>(_w_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_b_grads(num_threads, std::vector<double>(_b_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_rw_grads(num_threads, std::vector<double>(_rw_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_f_w_grads(num_threads, std::vector<double>(_f_w_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_f_b_grads(num_threads, std::vector<double>(_f_b_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_f_rw_grads(num_threads, std::vector<double>(_f_rw_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_i_w_grads(num_threads, std::vector<double>(_i_w_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_i_b_grads(num_threads, std::vector<double>(_i_b_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_i_rw_grads(num_threads, std::vector<double>(_i_rw_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_o_w_grads(num_threads, std::vector<double>(_o_w_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_o_b_grads(num_threads, std::vector<double>(_o_b_grads.size(), 0.0));
  std::vector<std::vector<double>> thread_o_rw_grads(num_threads, std::vector<double>(_o_rw_grads.size(), 0.0));

  auto run_chunk = [&](size_t start, size_t end, size_t thread_idx)
  {
    auto& local_w_grads = thread_w_grads[thread_idx];
    auto& local_b_grads = thread_b_grads[thread_idx];
    auto& local_rw_grads = thread_rw_grads[thread_idx];
    auto& local_f_w_grads = thread_f_w_grads[thread_idx];
    auto& local_f_b_grads = thread_f_b_grads[thread_idx];
    auto& local_f_rw_grads = thread_f_rw_grads[thread_idx];
    auto& local_i_w_grads = thread_i_w_grads[thread_idx];
    auto& local_i_b_grads = thread_i_b_grads[thread_idx];
    auto& local_i_rw_grads = thread_i_rw_grads[thread_idx];
    auto& local_o_w_grads = thread_o_w_grads[thread_idx];
    auto& local_o_b_grads = thread_o_b_grads[thread_idx];
    auto& local_o_rw_grads = thread_o_rw_grads[thread_idx];

    for (size_t b = start; b < end; ++b)
    {
      const auto& packed_grads = batch_gradients_and_outputs[b].get_rnn_gate_gradients(get_layer_index());
      if (packed_grads.empty()) continue;
      const auto& layer_states = hidden_states[b].at(get_layer_index());
      const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
      const auto& std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
      const double* x_base = !rnn_in.empty() ? rnn_in.data() : std_in.data();
      const size_t x_seq_len = !rnn_in.empty() ? rnn_in.size() / N_prev : 1;

      for (size_t t = 0; t < T; ++t)
      {
        const double* df = &packed_grads[t * GateCount * N_this];
        const double* di = &packed_grads[t * GateCount * N_this + N_this];
        const double* do_gate = &packed_grads[t * GateCount * N_this + 2 * N_this];
        const double* dg = &packed_grads[t * GateCount * N_this + 3 * N_this]; // Gate 4 (Candidate)
        const double* x_t = (x_seq_len == T) ? &x_base[t * N_prev] : x_base;
        const double* h_prev = (t > 0) ? layer_states[t - 1].get_hidden_state_values().data() : nullptr;

        for (size_t j = 0; j < N_this; ++j)
        {
          const double dfj = df[j], dij = di[j], doj = do_gate[j], dgj = dg[j];
          if (std::abs(dfj) < 1e-15 && std::abs(dij) < 1e-15 && std::abs(doj) < 1e-15 && std::abs(dgj) < 1e-15)
          {
            continue;
          }

          local_f_b_grads[j] += dfj; local_i_b_grads[j] += dij; local_o_b_grads[j] += doj; local_b_grads[j] += dgj;
          for (size_t k = 0; k < N_prev; ++k)
          {
            const double xk = x_t[k];
            if (std::abs(xk) < 1e-15)
            {
              continue;
            }
            local_f_w_grads[k * N_this + j] += dfj * xk;
            local_i_w_grads[k * N_this + j] += dij * xk;
            local_o_w_grads[k * N_this + j] += doj * xk;
            local_w_grads[k * N_this + j] += dgj * xk;
          }
          if (h_prev)
          {
            for (size_t k = 0; k < N_this; ++k)
            {
              const double hk = h_prev[k];
              if (std::abs(hk) < 1e-15)
              {
                continue;
              }
              local_f_rw_grads[k * N_this + j] += dfj * hk;
              local_i_rw_grads[k * N_this + j] += dij * hk;
              local_o_rw_grads[k * N_this + j] += doj * hk;
              local_rw_grads[k * N_this + j] += dgj * hk;
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
        _task_queue_pool->enqueue([start, end, t, &run_chunk]() { run_chunk(start, end, t); });
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  // Merge
  zero_gradients();
  for (unsigned int t = 0; t < num_threads; ++t)
  {
    for (size_t i = 0; i < _w_grads.size(); ++i) {
      _w_grads[i] += thread_w_grads[t][i]; _f_w_grads[i] += thread_f_w_grads[t][i];
      _i_w_grads[i] += thread_i_w_grads[t][i]; _o_w_grads[i] += thread_o_w_grads[t][i];
    }
    for (size_t i = 0; i < _rw_grads.size(); ++i) {
      _rw_grads[i] += thread_rw_grads[t][i]; _f_rw_grads[i] += thread_f_rw_grads[t][i];
      _i_rw_grads[i] += thread_i_rw_grads[t][i]; _o_rw_grads[i] += thread_o_rw_grads[t][i];
    }
    for (size_t i = 0; i < _b_grads.size(); ++i) {
      _b_grads[i] += thread_b_grads[t][i]; _f_b_grads[i] += thread_f_b_grads[t][i];
      _i_b_grads[i] += thread_i_b_grads[t][i]; _o_b_grads[i] += thread_o_b_grads[t][i];
    }
  }

  const double inv_batch = 1.0 / static_cast<double>(batch_size);
  auto norm = [&](std::vector<double>& v) { for (double& x : v) x *= inv_batch; };
  norm(_w_grads); norm(_b_grads); norm(_rw_grads);
  norm(_f_w_grads); norm(_f_b_grads); norm(_f_rw_grads);
  norm(_i_w_grads); norm(_i_b_grads); norm(_i_rw_grads);
  norm(_o_w_grads); norm(_o_b_grads); norm(_o_rw_grads);
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
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  auto app = [&](std::vector<double>& v, std::vector<double>& g, std::vector<double>& vel, std::vector<double>& m1, std::vector<double>& m2, std::vector<long long>& ts, const std::vector<double>& dec, bool is_bias) {
    apply_update_to_vector(v, g, vel, m1, m2, ts, dec, learning_rate, clipping_scale, is_bias, get_optimiser_type());
  };
  app(_f_w_values, _f_w_grads, _f_w_velocities, _f_w_m1, _f_w_m2, _f_w_timesteps, _f_w_decays, false);
  app(_f_b_values, _f_b_grads, _f_b_velocities, _f_b_m1, _f_b_m2, _f_b_timesteps, _f_b_decays, true);
  app(_f_rw_values, _f_rw_grads, _f_rw_velocities, _f_rw_m1, _f_rw_m2, _f_rw_timesteps, _f_rw_decays, false);
  app(_i_w_values, _i_w_grads, _i_w_velocities, _i_w_m1, _i_w_m2, _i_w_timesteps, _i_w_decays, false);
  app(_i_b_values, _i_b_grads, _i_b_velocities, _i_b_m1, _i_b_m2, _i_b_timesteps, _i_b_decays, true);
  app(_i_rw_values, _i_rw_grads, _i_rw_velocities, _i_rw_m1, _i_rw_m2, _i_rw_timesteps, _i_rw_decays, false);
  app(_w_values, _w_grads, _w_velocities, _w_m1, _w_m2, _w_timesteps, _w_decays, false);
  app(_b_values, _b_grads, _b_velocities, _b_m1, _b_m2, _b_timesteps, _b_decays, true);
  app(_rw_values, _rw_grads, _rw_velocities, _rw_m1, _rw_m2, _rw_timesteps, _rw_decays, false);
  app(_o_w_values, _o_w_grads, _o_w_velocities, _o_w_m1, _o_w_m2, _o_w_timesteps, _o_w_decays, false);
  app(_o_b_values, _o_b_grads, _o_b_velocities, _o_b_m1, _o_b_m2, _o_b_timesteps, _o_b_decays, true);
  app(_o_rw_values, _o_rw_grads, _o_rw_velocities, _o_rw_m1, _o_rw_m2, _o_rw_timesteps, _o_rw_decays, false);
  zero_gradients();
}

void LSTMLayer::cache_recurrent_weights()
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t n = get_number_neurons();
  if (n == 0) return;
  _rw_values_T.resize(n * n); _f_rw_values_T.resize(n * n); _i_rw_values_T.resize(n * n); _o_rw_values_T.resize(n * n);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j) {
      _rw_values_T[j * n + i] = _rw_values[i * n + j];
      _f_rw_values_T[j * n + i] = _f_rw_values[i * n + j];
      _i_rw_values_T[j * n + i] = _i_rw_values[i * n + j];
      _o_rw_values_T[j * n + i] = _o_rw_values[i * n + j];
    }
}

void LSTMLayer::allocate_workspace()
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  if (_task_queue_pool == nullptr)
  {
    return;
  }
  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  allocate_workspace(num_threads);
}

void LSTMLayer::allocate_workspace(unsigned int num_threads)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
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

LSTMLayer::BPTTWorkspace& LSTMLayer::get_workspace(size_t thread_idx) const
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
#if VALIDATE_DATA == 1
  if (thread_idx >= _thread_workspaces.size())
  {
    Logger::panic("Trying to get a workspace thread ", thread_idx, " past the workspaces size!");
  }
#endif
  return *_thread_workspaces[thread_idx];
}

double LSTMLayer::get_recurrent_weight_value(unsigned f, unsigned t) const
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  return _rw_values[f * get_number_neurons() + t];
}

void LSTMLayer::calculate_bptt_batch_chunk(size_t start, size_t end, std::vector<GradientsAndOutputs>& batch_gradients_and_outputs, const Layer& next_layer, const std::vector<std::vector<double>>& batch_next_grad_matrix, const std::vector<HiddenStates>& batch_hidden_states, int bptt_max_ticks, BPTTWorkspace& workspace, const BPTTWorkspace::AlignedVector& rw_values_T, const BPTTWorkspace::AlignedVector& f_rw_values_T, const BPTTWorkspace::AlignedVector& i_rw_values_T, const BPTTWorkspace::AlignedVector& o_rw_values_T) const
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  const int t_start = static_cast<int>(num_time_steps) - 1;
  int t_end = (bptt_max_ticks > 0) ? std::max(0, t_start - bptt_max_ticks + 1) : 0;
  workspace.resize(N_this, N_prev, end - start, num_time_steps);
  const size_t N_next = next_layer.get_number_neurons();
  const bool next_is_seq = (batch_next_grad_matrix[0].size() == num_time_steps * N_next);

  for (size_t b = start; b < end; ++b) {
    const size_t b_idx = b - start;
    const double* next_grads_base = batch_next_grad_matrix[b].data();
    double* dest_base = &workspace.grad_from_next_all_t[b_idx * num_time_steps * N_this];
    for (int t = t_start; t >= t_end; --t) {
      if (!next_is_seq && t < t_start) continue;
      const double* g_next_t = next_is_seq ? &next_grads_base[t * N_next] : next_grads_base;
      double* dest_t = &dest_base[t * N_this];
      for (size_t j = 0; j < N_this; ++j) dest_t[j] += simd::dot_product(g_next_t, next_layer.get_w_values().data() + j * N_next, N_next);
    }
  }

  for (int t = t_start; t >= t_end; --t) {
    for (size_t b = start; b < end; ++b) {
      const size_t b_idx = b - start;
      const auto& layer_states = batch_hidden_states[b].at(get_layer_index());
      const auto& state = layer_states[t];
      const auto packed = state.get_pre_activation_sums();
      const auto c_curr = state.get_cell_state_values();
      const bool has_prev = (t > 0);
      const auto c_prev = has_prev ? layer_states[t - 1].get_cell_state_values() : std::span<const double>();
      double* dh_next = &workspace.d_next_h[b_idx * N_this];
      double* dc_next = &workspace.d_next_c[b_idx * N_this];
      if (t == t_start) { std::fill(dh_next, dh_next + N_this, 0.0); std::fill(dc_next, dc_next + N_this, 0.0); }
      const double* upstream_grads = &workspace.grad_from_next_all_t[(b_idx * num_time_steps + t) * N_this];
      std::vector<double> dh_curr(N_this);

#if VALIDATE_DATA == 1
      if (packed.size() < Multiplier * N_this)
      {
        Logger::panic("LSTMLayer BPTT: HiddenState size mismatch! Expected multiplier ", Multiplier, ", but got ", (N_this > 0 ? packed.size() / N_this : 0));
      }
#endif

      for (size_t j = 0; j < N_this; ++j)
      {
        // Apply dropout mask stored during forward feed
        const double mask = packed[GateCount * N_this + j];
        dh_curr[j] = std::clamp((upstream_grads[j] + dh_next[j]) * mask, -50.0, 50.0);
      }

      double* df_chunk = &workspace.chunk_df[b_idx * N_this];
      double* di_chunk = &workspace.chunk_di[b_idx * N_this];
      double* do_chunk = &workspace.chunk_do[b_idx * N_this];
      double* dg_chunk = &workspace.chunk_dg[b_idx * N_this];

      simd::lstm_bptt_gate_step(N_this, dh_curr.data(), dc_next, packed.data(), &packed[N_this], &packed[2 * N_this], &packed[3 * N_this], c_curr.data(), c_prev.data(), has_prev, df_chunk, di_chunk, do_chunk, dg_chunk, dc_next);

      double* dx_t = &workspace.dx_matrix[(b_idx * num_time_steps + t) * N_prev];
      for (size_t k = 0; k < N_prev; ++k)
        dx_t[k] = simd::dot_product(df_chunk, &_f_w_values[k * N_this], N_this) + simd::dot_product(di_chunk, &_i_w_values[k * N_this], N_this) + simd::dot_product(do_chunk, &_o_w_values[k * N_this], N_this) + simd::dot_product(dg_chunk, &_w_values[k * N_this], N_this);

      std::fill(dh_next, dh_next + N_this, 0.0);
      for (size_t k = 0; k < N_this; ++k)
        dh_next[k] = simd::dot_product(df_chunk, &_f_rw_values[k * N_this], N_this) + simd::dot_product(di_chunk, &_i_rw_values[k * N_this], N_this) + simd::dot_product(do_chunk, &_o_rw_values[k * N_this], N_this) + simd::dot_product(dg_chunk, &_rw_values[k * N_this], N_this);

      double* grad_out_t = &workspace.rnn_grad_matrix[(b_idx * num_time_steps + t) * GateCount * N_this];
      std::copy(df_chunk, df_chunk + N_this, grad_out_t);
      std::copy(di_chunk, di_chunk + N_this, grad_out_t + N_this);
      std::copy(do_chunk, do_chunk + N_this, grad_out_t + 2 * N_this);
      std::copy(dg_chunk, dg_chunk + N_this, grad_out_t + 3 * N_this); // Gate 4 (Candidate)
    }
  }

  for (size_t b = start; b < end; ++b) {
    const size_t b_idx = b - start;
    const double* dX_src = &workspace.dx_matrix[b_idx * num_time_steps * N_prev];
    batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), std::vector<double>(dX_src, dX_src + num_time_steps * N_prev));
    const double* gates_src = &workspace.rnn_grad_matrix[b_idx * num_time_steps * GateCount * N_this];
    batch_gradients_and_outputs[b].set_rnn_gate_gradients(get_layer_index(), std::vector<double>(gates_src, gates_src + num_time_steps * GateCount * N_this));
  }
}

void LSTMLayer::set_w_values(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t expected_size = N_this * N_prev;
  if (v.size() == expected_size * GateCount)
  {
    _f_w_values.assign(v.begin(), v.begin() + expected_size);
    _i_w_values.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _o_w_values.assign(v.begin() + 2 * expected_size, v.begin() + 3 * expected_size);
    Layer::set_w_values(std::vector<double>(v.begin() + 3 * expected_size, v.end()));
  }
  else
  {
    Layer::set_w_values(v);
  }
}

void LSTMLayer::set_w_grads(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t expected_size = N_this * N_prev;
  if (v.size() == expected_size * GateCount)
  {
    _f_w_grads.assign(v.begin(), v.begin() + expected_size);
    _i_w_grads.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _o_w_grads.assign(v.begin() + 2 * expected_size, v.begin() + 3 * expected_size);
    Layer::set_w_grads(std::vector<double>(v.begin() + 3 * expected_size, v.end()));
  }
  else
  {
    Layer::set_w_grads(v);
  }
}

void LSTMLayer::set_w_velocities(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t expected_size = N_this * N_prev;
  if (v.size() == expected_size * GateCount)
  {
    _f_w_velocities.assign(v.begin(), v.begin() + expected_size);
    _i_w_velocities.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _o_w_velocities.assign(v.begin() + 2 * expected_size, v.begin() + 3 * expected_size);
    Layer::set_w_velocities(std::vector<double>(v.begin() + 3 * expected_size, v.end()));
  }
  else
  {
    Layer::set_w_velocities(v);
  }
}

void LSTMLayer::set_w_m1(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t expected_size = N_this * N_prev;
  if (v.size() == expected_size * GateCount)
  {
    _f_w_m1.assign(v.begin(), v.begin() + expected_size);
    _i_w_m1.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _o_w_m1.assign(v.begin() + 2 * expected_size, v.begin() + 3 * expected_size);
    Layer::set_w_m1(std::vector<double>(v.begin() + 3 * expected_size, v.end()));
  }
  else
  {
    Layer::set_w_m1(v);
  }
}

void LSTMLayer::set_w_m2(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t expected_size = N_this * N_prev;
  if (v.size() == expected_size * GateCount)
  {
    _f_w_m2.assign(v.begin(), v.begin() + expected_size);
    _i_w_m2.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _o_w_m2.assign(v.begin() + 2 * expected_size, v.begin() + 3 * expected_size);
    Layer::set_w_m2(std::vector<double>(v.begin() + 3 * expected_size, v.end()));
  }
  else
  {
    Layer::set_w_m2(v);
  }
}

void LSTMLayer::set_w_timesteps(const std::vector<long long>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t expected_size = N_this * N_prev;
  if (v.size() == expected_size * GateCount)
  {
    _f_w_timesteps.assign(v.begin(), v.begin() + expected_size);
    _i_w_timesteps.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _o_w_timesteps.assign(v.begin() + 2 * expected_size, v.begin() + 3 * expected_size);
    Layer::set_w_timesteps(std::vector<long long>(v.begin() + 3 * expected_size, v.end()));
  }
  else
  {
    Layer::set_w_timesteps(v);
  }
}

void LSTMLayer::set_w_decays(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t expected_size = N_this * N_prev;
  if (v.size() == expected_size * GateCount)
  {
    _f_w_decays.assign(v.begin(), v.begin() + expected_size);
    _i_w_decays.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _o_w_decays.assign(v.begin() + 2 * expected_size, v.begin() + 3 * expected_size);
    Layer::set_w_decays(std::vector<double>(v.begin() + 3 * expected_size, v.end()));
  }
  else
  {
    Layer::set_w_decays(v);
  }
}

void LSTMLayer::set_b_values(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  if (v.size() == N_this * GateCount)
  {
    _f_b_values.assign(v.begin(), v.begin() + N_this);
    _i_b_values.assign(v.begin() + N_this, v.begin() + 2 * N_this);
    _o_b_values.assign(v.begin() + 2 * N_this, v.begin() + 3 * N_this);
    Layer::set_b_values(std::vector<double>(v.begin() + 3 * N_this, v.end()));
  }
  else
  {
    Layer::set_b_values(v);
  }
}

void LSTMLayer::set_b_grads(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  if (v.size() == N_this * GateCount)
  {
    _f_b_grads.assign(v.begin(), v.begin() + N_this);
    _i_b_grads.assign(v.begin() + N_this, v.begin() + 2 * N_this);
    _o_b_grads.assign(v.begin() + 2 * N_this, v.begin() + 3 * N_this);
    Layer::set_b_grads(std::vector<double>(v.begin() + 3 * N_this, v.end()));
  }
  else
  {
    Layer::set_b_grads(v);
  }
}

void LSTMLayer::set_b_velocities(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  if (v.size() == N_this * GateCount)
  {
    _f_b_velocities.assign(v.begin(), v.begin() + N_this);
    _i_b_velocities.assign(v.begin() + N_this, v.begin() + 2 * N_this);
    _o_b_velocities.assign(v.begin() + 2 * N_this, v.begin() + 3 * N_this);
    Layer::set_b_velocities(std::vector<double>(v.begin() + 3 * N_this, v.end()));
  }
  else
  {
    Layer::set_b_velocities(v);
  }
}

void LSTMLayer::set_b_m1(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  if (v.size() == N_this * GateCount)
  {
    _f_b_m1.assign(v.begin(), v.begin() + N_this);
    _i_b_m1.assign(v.begin() + N_this, v.begin() + 2 * N_this);
    _o_b_m1.assign(v.begin() + 2 * N_this, v.begin() + 3 * N_this);
    Layer::set_b_m1(std::vector<double>(v.begin() + 3 * N_this, v.end()));
  }
  else
  {
    Layer::set_b_m1(v);
  }
}

void LSTMLayer::set_b_m2(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  if (v.size() == N_this * GateCount)
  {
    _f_b_m2.assign(v.begin(), v.begin() + N_this);
    _i_b_m2.assign(v.begin() + N_this, v.begin() + 2 * N_this);
    _o_b_m2.assign(v.begin() + 2 * N_this, v.begin() + 3 * N_this);
    Layer::set_b_m2(std::vector<double>(v.begin() + 3 * N_this, v.end()));
  }
  else
  {
    Layer::set_b_m2(v);
  }
}

void LSTMLayer::set_b_timesteps(const std::vector<long long>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  if (v.size() == N_this * GateCount)
  {
    _f_b_timesteps.assign(v.begin(), v.begin() + N_this);
    _i_b_timesteps.assign(v.begin() + N_this, v.begin() + 2 * N_this);
    _o_b_timesteps.assign(v.begin() + 2 * N_this, v.begin() + 3 * N_this);
    Layer::set_b_timesteps(std::vector<long long>(v.begin() + 3 * N_this, v.end()));
  }
  else
  {
    Layer::set_b_timesteps(v);
  }
}

void LSTMLayer::set_b_decays(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  if (v.size() == N_this * GateCount)
  {
    _f_b_decays.assign(v.begin(), v.begin() + N_this);
    _i_b_decays.assign(v.begin() + N_this, v.begin() + 2 * N_this);
    _o_b_decays.assign(v.begin() + 2 * N_this, v.begin() + 3 * N_this);
    Layer::set_b_decays(std::vector<double>(v.begin() + 3 * N_this, v.end()));
  }
  else
  {
    Layer::set_b_decays(v);
  }
}

void LSTMLayer::set_rw_values(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t expected_size = N_this * N_this;
  if (v.size() == expected_size * GateCount)
  {
    _f_rw_values.assign(v.begin(), v.begin() + expected_size);
    _i_rw_values.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _o_rw_values.assign(v.begin() + 2 * expected_size, v.begin() + 3 * expected_size);
    _rw_values.assign(v.begin() + 3 * expected_size, v.end());
  }
  else
  {
    _rw_values = v;
  }
}

void LSTMLayer::set_rw_grads(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t expected_size = N_this * N_this;
  if (v.size() == expected_size * GateCount)
  {
    _f_rw_grads.assign(v.begin(), v.begin() + expected_size);
    _i_rw_grads.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _o_rw_grads.assign(v.begin() + 2 * expected_size, v.begin() + 3 * expected_size);
    _rw_grads.assign(v.begin() + 3 * expected_size, v.end());
  }
  else
  {
    _rw_grads = v;
  }
}

void LSTMLayer::set_rw_velocities(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t expected_size = N_this * N_this;
  if (v.size() == expected_size * GateCount)
  {
    _f_rw_velocities.assign(v.begin(), v.begin() + expected_size);
    _i_rw_velocities.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _o_rw_velocities.assign(v.begin() + 2 * expected_size, v.begin() + 3 * expected_size);
    _rw_velocities.assign(v.begin() + 3 * expected_size, v.end());
  }
  else
  {
    _rw_velocities = v;
  }
}

void LSTMLayer::set_rw_m1(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t expected_size = N_this * N_this;
  if (v.size() == expected_size * GateCount)
  {
    _f_rw_m1.assign(v.begin(), v.begin() + expected_size);
    _i_rw_m1.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _o_rw_m1.assign(v.begin() + 2 * expected_size, v.begin() + 3 * expected_size);
    _rw_m1.assign(v.begin() + 3 * expected_size, v.end());
  }
  else
  {
    _rw_m1 = v;
  }
}

void LSTMLayer::set_rw_m2(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t expected_size = N_this * N_this;
  if (v.size() == expected_size * GateCount)
  {
    _f_rw_m2.assign(v.begin(), v.begin() + expected_size);
    _i_rw_m2.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _o_rw_m2.assign(v.begin() + 2 * expected_size, v.begin() + 3 * expected_size);
    _rw_m2.assign(v.begin() + 3 * expected_size, v.end());
  }
  else
  {
    _rw_m2 = v;
  }
}

void LSTMLayer::set_rw_timesteps(const std::vector<long long>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t expected_size = N_this * N_this;
  if (v.size() == expected_size * GateCount)
  {
    _f_rw_timesteps.assign(v.begin(), v.begin() + expected_size);
    _i_rw_timesteps.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _o_rw_timesteps.assign(v.begin() + 2 * expected_size, v.begin() + 3 * expected_size);
    _rw_timesteps.assign(v.begin() + 3 * expected_size, v.end());
  }
  else
  {
    _rw_timesteps = v;
  }
}

void LSTMLayer::set_rw_decays(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
  const size_t N_this = get_number_neurons();
  const size_t expected_size = N_this * N_this;
  if (v.size() == expected_size * GateCount)
  {
    _f_rw_decays.assign(v.begin(), v.begin() + expected_size);
    _i_rw_decays.assign(v.begin() + expected_size, v.begin() + 2 * expected_size);
    _o_rw_decays.assign(v.begin() + 2 * expected_size, v.begin() + 3 * expected_size);
    _rw_decays.assign(v.begin() + 3 * expected_size, v.end());
  }
  else
  {
    _rw_decays = v;
  }
}
