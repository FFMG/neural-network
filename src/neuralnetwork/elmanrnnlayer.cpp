#include "./libraries/instrumentor.h"
#include "elmanrnnlayer.h"
#include "logger.h"

#include <iostream>
#include <numeric>

constexpr bool _has_bias_neuron = true;

ElmanRNNLayer::ElmanRNNLayer(
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
  )
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  initialize_recurrent_weights(weight_decay);
}

ElmanRNNLayer::ElmanRNNLayer(const ElmanRNNLayer& src) noexcept :
  Layer(src),
  _rw_values(src._rw_values),
  _rw_grads(src._rw_grads),
  _rw_velocities(src._rw_velocities),
  _rw_m1(src._rw_m1),
  _rw_m2(src._rw_m2),
  _rw_timesteps(src._rw_timesteps),
  _rw_decays(src._rw_decays),
  _residual_weights(src._residual_weights)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
}

ElmanRNNLayer::ElmanRNNLayer(ElmanRNNLayer&& src) noexcept :
  Layer(std::move(src)),
  _rw_values(std::move(src._rw_values)),
  _rw_grads(std::move(src._rw_grads)),
  _rw_velocities(std::move(src._rw_velocities)),
  _rw_m1(std::move(src._rw_m1)),
  _rw_m2(std::move(src._rw_m2)),
  _rw_timesteps(std::move(src._rw_timesteps)),
  _rw_decays(std::move(src._rw_decays)),
  _residual_weights(std::move(src._residual_weights))
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
}

ElmanRNNLayer& ElmanRNNLayer::operator=(const ElmanRNNLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
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
    _residual_weights = src._residual_weights;
  }
  return *this;
}

ElmanRNNLayer& ElmanRNNLayer::operator=(ElmanRNNLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
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
    _residual_weights = std::move(src._residual_weights);
  }
  return *this;
}

ElmanRNNLayer::~ElmanRNNLayer() = default;

void ElmanRNNLayer::initialize_recurrent_weights(double weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const auto num_neurons = get_number_output_neurons();
  const size_t num_weights = static_cast<size_t>(num_neurons) * num_neurons;

  auto initial_weights = get_activation().weight_initialization(num_neurons, num_neurons);
  _rw_values.resize(num_weights);
  for (size_t i = 0; i < num_neurons; ++i) {
    for (size_t o = 0; o < num_neurons; ++o) {
        _rw_values[i * num_neurons + o] = initial_weights[o];
    }
  }

  _rw_grads.assign(num_weights, 0.0);
  _rw_velocities.assign(num_weights, 0.0);
  _rw_m1.assign(num_weights, 0.0);
  _rw_m2.assign(num_weights, 0.0);
  _rw_timesteps.assign(num_weights, 0);
  _rw_decays.assign(num_weights, weight_decay);
}

bool ElmanRNNLayer::has_bias() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  return _has_bias_neuron;
}

std::vector<double> ElmanRNNLayer::calculate_forward_feed(
  GradientsAndOutputs& gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<double>& previous_layer_inputs,
  const std::vector<double>& residual_output_values,
  std::vector<HiddenState>& hidden_states,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t N_prev = previous_layer.get_number_neurons();
  const size_t N_this = get_number_neurons();
  const size_t num_time_steps = N_prev > 0 ? previous_layer_inputs.size() / N_prev : 0;

  std::vector<double> output_sequence(num_time_steps * N_this, 0.0);
  std::vector<double> last_output_sequence(N_this, 0.0);
    
  std::vector<double> current_input_t(N_prev);
  std::vector<double> pre_activation_sums(N_this);
    
  std::fill(pre_activation_sums.begin(), pre_activation_sums.end(), 0.0);
    
  // hidden_states is already sized for a single batch item, so no need to check empty()
  // and hidden_states.assign (it's passed by reference and managed by NeuralNetwork)
  assert(hidden_states.size() == num_time_steps);
  std::vector<double> prev_hidden_state_values(N_this, 0.0);
  for (size_t t = 0; t < num_time_steps; ++t) 
  {
    // Populate current_input_t
    
    for(size_t i = 0; i < N_prev; ++i) 
    {
      current_input_t[i] = previous_layer_inputs[t * N_prev + i];
    }
    
    std::fill(pre_activation_sums.begin(), pre_activation_sums.end(), 0.0);
    if (has_bias())
    {
        for (size_t j = 0; j < N_this; ++j) 
        {
            pre_activation_sums[j] = get_bias_value((unsigned)j);
        }
    }

    if (get_layer_type() != LayerType::Input) 
    {
      for (size_t i = 0; i < N_prev; ++i) 
      {
        for (size_t j = 0; j < N_this; ++j) 
        {
          pre_activation_sums[j] += current_input_t[i] * get_weight_value((unsigned)i, (unsigned)j);
        }
      }
    }
        
    if (get_layer_type() == LayerType::Hidden || get_layer_type() == LayerType::Output)
    {
      // Only apply recurrent contribution from the previous time-step.
      // Use the explicit stored hidden state for t-1 to avoid relying on
      // local prev_hidden_state_values and to make indexing explicit.
      if (t > 0)
      {
        // small debug snapshot before adding recurrent contribution
        Logger::trace([=] 
        {
          return Logger::factory("RNN Forward t=", t, " using hidden_states[t-1] sample h0=",
            hidden_states[static_cast<size_t>(t - 1)].get_hidden_state_value_at_neuron(0),
            ", h1=",
            (N_this > 1 ? hidden_states[static_cast<size_t>(t - 1)].get_hidden_state_value_at_neuron(1) : 0.0));
          });

        for (size_t i = 0; i < N_this; ++i) 
        {
          // get previous hidden value explicitly from hidden_states[t-1]
          const double h_prev_i = hidden_states[static_cast<size_t>(t - 1)].get_hidden_state_value_at_neuron(static_cast<unsigned>(i));
          if (h_prev_i == 0.0) 
          {
            continue; // minor micro-optimisation and avoids noisy adds
          }
          for (size_t j = 0; j < N_this; ++j) 
          {
            pre_activation_sums[j] += h_prev_i * get_recurrent_weight_value((unsigned)i, (unsigned)j);
          }
        }

        // snapshot after adding recurrent contribution
        Logger::trace([=]
          {
          return Logger::factory("RNN Forward t=", t, " pre_act_sample[0]=",
            pre_activation_sums.size() ? pre_activation_sums[0] : 0.0,
            ", [1]=",
            pre_activation_sums.size() > 1 ? pre_activation_sums[1] : 0.0);
          });
      }
      else
      {
        // t == 0: initial hidden state is assumed zero � explicit log helps debugging
        Logger::trace([=] 
        {
          return Logger::factory("RNN Forward t=0: skipping recurrent add (initial hidden state = 0).");
          });
      }
    }

	  if (!residual_output_values.empty())
	  {
		  // Use the layer's residual projector if present
		  if (const ResidualProjector* rp = get_residual_projector())
		  {
			  auto projected = rp->project(residual_output_values); // size == N_this (output size)
			  // add projected residual contribution to pre-activation sums
			  for (size_t j = 0; j < N_this; ++j)
			  {
				  pre_activation_sums[j] += projected[j];
			  }
			  if (Logger::can_trace()) {
				  Logger::trace([=] { return Logger::factory("RNN Forward t=", t, " added residual projection sample=", pre_activation_sums.size()?pre_activation_sums[0]:0.0); });
			  }
		  }
	  }

    std::vector<double> current_hidden_state_values(N_this);
    for (size_t j = 0; j < N_this; ++j) 
    {
      const auto& neuron = get_neuron((unsigned)j);
      double output = get_activation().activate(pre_activation_sums[j]);

      if (is_training && neuron.is_dropout()) 
      {
        if (neuron.must_randomly_drop()) 
        {
          output = 0.0;
        } 
        else 
        {
          output /= (1.0 - neuron.get_dropout_rate());
        }
      }
      current_hidden_state_values[j] = output;
      output_sequence[t * N_this + j] = output;
      if(t == num_time_steps - 1)
      {
          last_output_sequence[j] = output;
      }
    }
        
    hidden_states[t].set_pre_activation_sums(pre_activation_sums);
    hidden_states[t].set_hidden_state_values(current_hidden_state_values);
    prev_hidden_state_values = current_hidden_state_values;
  }
  gradients_and_outputs.set_rnn_outputs(get_layer_index(), output_sequence);
  gradients_and_outputs.set_outputs(get_layer_index(), last_output_sequence);
  return last_output_sequence;
}

void ElmanRNNLayer::calculate_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs,
  ErrorCalculation::type error_calculation_type) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  switch (error_calculation_type)
  {
  case ErrorCalculation::type::mse:
    return calculate_mse_error_deltas(deltas, target_outputs, given_outputs);
  case ErrorCalculation::type::bce_loss:
    return calculate_bce_error_deltas(deltas, target_outputs, given_outputs);
  default:
    Logger::panic("ErrorCalculation type is not supported for ElmanRNNLayer!");
  }
}

void ElmanRNNLayer::calculate_bce_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t N_total = get_number_neurons();
  const double denom = static_cast<double>(N_total);

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    deltas[neuron_index] = (given_outputs[neuron_index] - target_outputs[neuron_index]) / denom;
  }
}

void ElmanRNNLayer::calculate_mse_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t N_total = get_number_neurons();
  const double denom = static_cast<double>(N_total);

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    deltas[neuron_index] = (given_outputs[neuron_index] - target_outputs[neuron_index]) / denom;
  }
}

void ElmanRNNLayer::calculate_output_gradients(
  GradientsAndOutputs& gradients_and_outputs,
  const std::vector<double>& target_outputs,
  const std::vector<HiddenState>& hidden_states,
  ErrorCalculation::type error_calculation_type) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t N_total = get_number_neurons();

  // Get the outputs stored for this layer. For RNNs this is a sequence:
  // given_outputs.size() == num_time_steps * N_total
  const auto& given_outputs = gradients_and_outputs.get_outputs(get_layer_index());

  // Prepare output gradients (one value per neuron, not per time-step)
  std::vector<double> gradients(N_total, 0.0);

  if (given_outputs.size() == N_total)
  {
    // Non-RNN (single timestep) case
    std::vector<double> deltas(N_total, 0.0);
    calculate_error_deltas(deltas, target_outputs, given_outputs, error_calculation_type);

    // hidden_states is expected to contain a single HiddenState for output layer
    const HiddenState* hs_ptr = nullptr;
    if (!hidden_states.empty())
    {
      hs_ptr = &hidden_states.back();
    }

    if (error_calculation_type == ErrorCalculation::type::bce_loss && get_activation().get_method() == activation::method::sigmoid)
    {
		  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
		  {
			  gradients[neuron_index] = deltas[neuron_index];
		  }
    }
    else
    {
		  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
		  {
		    double deriv = 1.0;
		    if (hs_ptr != nullptr)
		    {
  			  deriv = get_activation().activate_derivative(hs_ptr->get_pre_activation_sum_at_neuron(neuron_index));
		    }
		    gradients[neuron_index] = deltas[neuron_index] * deriv;
		  }
    }
  }
  else if (given_outputs.size() >= N_total && N_total > 0)
  {
    // RNN case: treat outputs as a time sequence. For supervised targets that
    // represent the final output only, use the last time-step's outputs.
    const size_t num_time_steps = given_outputs.size() / N_total;
    if (num_time_steps == 0)
    {
      // defensive: nothing to do
      gradients_and_outputs.set_gradients(get_layer_index(), gradients);
      return;
    }

    // hidden_states should contain one HiddenState per time-step; use last time-step
    const HiddenState* last_hs_ptr = nullptr;
    if (!hidden_states.empty())
    {
      last_hs_ptr = &hidden_states.back();
    }

    if (error_calculation_type == ErrorCalculation::type::bce_loss && get_activation().get_method() == activation::method::sigmoid)
    {
		  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
		  {
		    const size_t last_idx = (num_time_steps - 1) * N_total + neuron_index;
		    const double last_output = given_outputs[last_idx];
		    const double target = (neuron_index < target_outputs.size()) ? target_outputs[neuron_index] : 0.0;
		    gradients[neuron_index] = (last_output - target) / static_cast<double>(N_total);
		  }
    }
    else
    {
		  // Compute gradient for each neuron using last time-step output and its pre-activation derivative
		  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
		  {
		    const size_t last_idx = (num_time_steps - 1) * N_total + neuron_index;
		    const double last_output = given_outputs[last_idx];
		    const double target = (neuron_index < target_outputs.size()) ? target_outputs[neuron_index] : 0.0;
		    const double delta = (last_output - target) / static_cast<double>(N_total);

		    double deriv = 1.0;
		    if (last_hs_ptr != nullptr)
		    {
			    deriv = get_activation().activate_derivative(last_hs_ptr->get_pre_activation_sum_at_neuron(neuron_index));
		    }
		    gradients[neuron_index] = delta * deriv;
		  }
    }
  }
  else
  {
    // Unexpected shape: fall back to zeros (defensive)
    for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
    {
      gradients[neuron_index] = 0.0;
    }
  }

  gradients_and_outputs.set_gradients(get_layer_index(), gradients);
}

void ElmanRNNLayer::calculate_hidden_gradients(
  GradientsAndOutputs& gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<double>& next_grad_matrix,
  const std::vector<double>& /*output_matrix*/,
  const std::vector<HiddenState>& hidden_states,
  int bptt_max_ticks) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");

  const size_t num_time_steps = hidden_states.size();
  const size_t N_this = get_number_neurons();
  const size_t N_next = next_layer.get_number_output_neurons();

  // Defensive: nothing to do
  if (num_time_steps == 0 || N_this == 0)
  {
    gradients_and_outputs.set_gradients(get_layer_index(), std::vector<double>(N_this, 0.0));
    gradients_and_outputs.set_rnn_gradients(get_layer_index(), std::vector<double>());
    return;
  }

  // Determine whether next_gradients are time-distributed or single-last-step
  const bool next_is_time_distributed = (next_grad_matrix.size() == N_next * num_time_steps);
  const bool next_is_last_only = (next_grad_matrix.size() == N_next);

  // Prepare accumulators
  std::vector<double> grad_matrix(N_this, 0.0);                      // aggregated dE/dz over time
  std::vector<double> rnn_grad_matrix(num_time_steps * N_this, 0.0); // per-time dE/dz
  std::vector<double> d_next_h(N_this, 0.0);                         // dE/dh(t) from t+1

  // Backprop through time
  const int t_start = static_cast<int>(num_time_steps) - 1;
  int t_end = 0;
  if (bptt_max_ticks > 0)
  {
    t_end = std::max(0, t_start - bptt_max_ticks + 1);
  }

  for (int t = t_start; t >= t_end; --t)
  {
    // 1) contribution from the "next" layer at time t (either time-distributed or last-only)
    std::vector<double> grad_from_next_layer(N_this, 0.0);
    if (next_is_time_distributed)
    {
      // slice for time t: next_grad_matrix[t * N_next + k]
      const size_t base = static_cast<size_t>(t) * N_next;
      for (size_t k = 0; k < N_next; ++k)
      {
        const double next_grad_val = next_grad_matrix[base + k];
        if (!std::isfinite(next_grad_val)) continue;
        // accumulate into each neuron i of this layer via next layer's input weight from i -> k
        for (size_t i = 0; i < N_this; ++i)
        {
          grad_from_next_layer[i] += next_grad_val * next_layer.get_weight_value(static_cast<unsigned>(i), static_cast<unsigned>(k));
        }
      }
    }
    else if (next_is_last_only)
    {
      // Only add next-layer contribution at final timestep
      if (t == t_start)
      {
        for (size_t k = 0; k < N_next; ++k)
        {
          const double next_grad_val = next_grad_matrix[k];
          if (!std::isfinite(next_grad_val)) continue;
          for (size_t i = 0; i < N_this; ++i)
          {
            grad_from_next_layer[i] += next_grad_val * next_layer.get_weight_value(static_cast<unsigned>(i), static_cast<unsigned>(k));
          }
        }
      }
    }
    else
    {
      // no next-layer contribution (shapes unexpected) — leave zeros but warn in trace
      if (Logger::can_trace())
      {
        Logger::trace([=] { return Logger::factory("HiddenGradients: next_grad_matrix shape unexpected (size=", next_grad_matrix.size(), ", expected ", N_next, " or ", N_next * num_time_steps, ")"); });
      }
    }

    // 2) compute per-time gradients into pre-activation z(t)
    for (size_t i = 0; i < N_this; ++i)
    {
      const double upstream = grad_from_next_layer[i] + d_next_h[i];
      const double preact = hidden_states[static_cast<size_t>(t)].get_pre_activation_sum_at_neuron(static_cast<unsigned>(i));
      const double deriv = get_activation().activate_derivative(preact);
      double g = upstream * deriv;
      if (!std::isfinite(g)) g = 0.0;
      // store raw per-time gradient (no per-element clipping here)
      rnn_grad_matrix[static_cast<size_t>(t) * N_this + i] = g;
      grad_matrix[i] += g; // accumulate over time
    }

    // 3) propagate through recurrent weights to form d_next_h for previous timestep
    std::fill(d_next_h.begin(), d_next_h.end(), 0.0);
    for (size_t j = 0; j < N_this; ++j)
    {
      const double g_j = rnn_grad_matrix[static_cast<size_t>(t) * N_this + j];
      if (g_j == 0.0) continue;
      for (size_t i = 0; i < N_this; ++i)
      {
        d_next_h[i] += g_j * get_recurrent_weight_value((unsigned)i, (unsigned)j);
      }
    }
  }

  // store gradients for later weight-gradient computation
  gradients_and_outputs.set_gradients(get_layer_index(), grad_matrix);
  gradients_and_outputs.set_rnn_gradients(get_layer_index(), rnn_grad_matrix);

  if (Logger::can_trace())
  {
    double norm = 0.0;
    double maxabs = 0.0;
    for (auto v : grad_matrix) { norm += v * v; maxabs = std::max(maxabs, std::fabs(v)); }
    norm = std::sqrt(norm);
    Logger::trace([=] { return Logger::factory("HiddenGradients END layer=", get_layer_index(), " grad_norm=", norm, " maxabs=", maxabs); });
  }
}

void ElmanRNNLayer::apply_recurrent_weight_gradient(unsigned from_neuron, unsigned to_neuron, double gradient, double learning_rate, double clipping_scale)
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

double ElmanRNNLayer::get_recurrent_weight_value(unsigned from_neuron, unsigned to_neuron) const
{
    return _rw_values[from_neuron * get_number_neurons() + to_neuron];
}

Layer* ElmanRNNLayer::clone() const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  return new ElmanRNNLayer(*this);
}

const std::vector<std::vector<WeightParam>>& ElmanRNNLayer::get_residual_weight_params() const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  return _residual_weights;
}

std::vector<std::vector<WeightParam>>& ElmanRNNLayer::get_residual_weight_params()
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  return _residual_weights;
}
