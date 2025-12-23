#include "./libraries/instrumentor.h"
#include "elmanrnnlayer.h"
#include "logger.h"

#include <iostream>
#include <numeric>
#include <future>
#include <thread>
#include <algorithm>

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
  _rw_decays(src._rw_decays)
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
  _rw_decays(std::move(src._rw_decays))
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

void ElmanRNNLayer::calculate_forward_feed(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t batch_size = batch_gradients_and_outputs.size();
  const size_t N_prev = previous_layer.get_number_neurons();
  const size_t N_this = get_number_neurons();

  std::vector<double> sample_inputs = batch_gradients_and_outputs[0].get_rnn_outputs(previous_layer.get_layer_index());
  if (sample_inputs.empty())
  {
      sample_inputs = batch_gradients_and_outputs[0].get_outputs(previous_layer.get_layer_index());
  }
  const size_t num_time_steps = N_prev > 0 ? sample_inputs.size() / N_prev : 0;

  std::vector<std::vector<double>> batch_output_sequences(batch_size, std::vector<double>(num_time_steps * N_this, 0.0));
  std::vector<std::vector<double>> batch_last_output_sequences(batch_size, std::vector<double>(N_this, 0.0));

  auto run_forward_pass = [&](size_t start, size_t end)
  {
    for (size_t b = start; b < end; ++b)
    {
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        std::vector<double> pre_activation_sums(N_this, 0.0);
        
        if (has_bias())
        {
          for (size_t j = 0; j < N_this; ++j)
          {
            pre_activation_sums[j] = get_bias_value((unsigned)j);
          }
        }

        if (get_layer_type() != LayerType::Input)
        {
          std::vector<double> prev_inputs = batch_gradients_and_outputs[b].get_rnn_outputs(previous_layer.get_layer_index());
          if (prev_inputs.empty()) prev_inputs = batch_gradients_and_outputs[b].get_outputs(previous_layer.get_layer_index());
          
          for (size_t i = 0; i < N_prev; ++i)
          {
            double val = (prev_inputs.size() == N_prev) ? prev_inputs[i] : (prev_inputs.size() >= (t + 1) * N_prev ? prev_inputs[t * N_prev + i] : 0.0);
            if (val == 0.0) continue;
            for (size_t j = 0; j < N_this; ++j)
            {
              pre_activation_sums[j] += val * get_weight_value((unsigned)i, (unsigned)j);
            }
          }
        }

        if (t > 0 && (get_layer_type() == LayerType::Hidden || get_layer_type() == LayerType::Output))
        {
          for (size_t i = 0; i < N_this; ++i)
          {
            const double h_prev_i = batch_hidden_states[b].at(get_layer_index())[t - 1].get_hidden_state_value_at_neuron((unsigned)i);
            if (h_prev_i == 0.0) continue;
            for (size_t j = 0; j < N_this; ++j)
            {
              pre_activation_sums[j] += h_prev_i * get_recurrent_weight_value((unsigned)i, (unsigned)j);
            }
          }
        }

        if (!batch_residual_output_values.empty())
        {
          if (batch_residual_output_values[b].size() == N_this)
          {
            for (size_t j = 0; j < N_this; ++j)
            {
              pre_activation_sums[j] += batch_residual_output_values[b][j];
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
            if (neuron.must_randomly_drop()) output = 0.0;
            else output /= (1.0 - neuron.get_dropout_rate());
          }
          current_hidden_state_values[j] = output;
          batch_output_sequences[b][t * N_this + j] = output;
          if (t == num_time_steps - 1) batch_last_output_sequences[b][j] = output;
        }
        batch_hidden_states[b].at(get_layer_index())[t].set_pre_activation_sums(pre_activation_sums);
        batch_hidden_states[b].at(get_layer_index())[t].set_hidden_state_values(current_hidden_state_values);
      }
    }
  };

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 2;
  if (batch_size < num_threads * 2) num_threads = 1;

  if (num_threads <= 1)
  {
    run_forward_pass(0, batch_size);
  }
  else
  {
    std::vector<std::future<void>> futures;
    size_t chunk_size = batch_size / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t start = t * chunk_size;
      size_t end = (t == num_threads - 1) ? batch_size : start + chunk_size;
      futures.push_back(std::async(std::launch::async, run_forward_pass, start, end));
    }
    for (auto& f : futures) f.get();
  }

  for (size_t b = 0; b < batch_size; ++b)
  {
    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), batch_output_sequences[b]);
    batch_gradients_and_outputs[b].set_outputs(get_layer_index(), batch_last_output_sequences[b]);
  }
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
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  ErrorCalculation::type error_calculation_type) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t batch_size = batch_gradients_and_outputs.size();
  const size_t N_total = get_number_neurons();

  auto run_output_gradients = [&](size_t start, size_t end)
  {
    for (size_t b = start; b < end; b++)
    {
      const auto& given_outputs = batch_gradients_and_outputs[b].get_outputs(get_layer_index());
      const auto& target_outputs = *(target_outputs_begin + b);
      std::vector<double> gradients(N_total, 0.0);

      if (given_outputs.size() == N_total)
      {
        std::vector<double> deltas(N_total, 0.0);
        calculate_error_deltas(deltas, target_outputs, given_outputs, error_calculation_type);
        const auto& last_hs = batch_hidden_states[b].at(get_layer_index()).back();

        if (error_calculation_type == ErrorCalculation::type::bce_loss && get_activation().get_method() == activation::method::sigmoid)
        {
          for (unsigned j = 0; j < N_total; ++j) gradients[j] = deltas[j];
        }
        else
        {
          for (unsigned j = 0; j < N_total; ++j)
          {
            double deriv = get_activation().activate_derivative(last_hs.get_pre_activation_sum_at_neuron(j));
            gradients[j] = deltas[j] * deriv;
          }
        }
      }
      else if (given_outputs.size() >= N_total && N_total > 0)
      {
        const size_t num_time_steps = given_outputs.size() / N_total;
        const auto& last_hs = batch_hidden_states[b].at(get_layer_index()).back();

        for (unsigned j = 0; j < N_total; ++j)
        {
          const size_t last_idx = (num_time_steps - 1) * N_total + j;
          const double target = (j < target_outputs.size()) ? target_outputs[j] : 0.0;
          const double delta = (given_outputs[last_idx] - target) / static_cast<double>(N_total);

          if (error_calculation_type == ErrorCalculation::type::bce_loss && get_activation().get_method() == activation::method::sigmoid)
          {
            gradients[j] = delta;
          }
          else
          {
            double deriv = get_activation().activate_derivative(last_hs.get_pre_activation_sum_at_neuron(j));
            gradients[j] = delta * deriv;
          }
        }
      }
      batch_gradients_and_outputs[b].set_gradients(get_layer_index(), gradients);
    }
  };

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 2;
  if (batch_size < num_threads * 2) num_threads = 1;

  if (num_threads <= 1)
  {
    run_output_gradients(0, batch_size);
  }
  else
  {
    std::vector<std::future<void>> futures;
    size_t chunk_size = batch_size / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t start = t * chunk_size;
      size_t end = (t == num_threads - 1) ? batch_size : start + chunk_size;
      futures.push_back(std::async(std::launch::async, run_output_gradients, start, end));
    }
    for (auto& f : futures) f.get();
  }
}

void ElmanRNNLayer::calculate_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<std::vector<double>>& batch_next_grad_matrix,
  const std::vector<HiddenStates>& batch_hidden_states,
  int bptt_max_ticks) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t batch_size = batch_gradients_and_outputs.size();
  const size_t N_this = get_number_neurons();
  const size_t N_next = next_layer.get_number_output_neurons();

  // Assuming all items have the same num_time_steps
  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  if (num_time_steps == 0 || N_this == 0)
  {
    for (size_t b = 0; b < batch_size; ++b)
    {
      batch_gradients_and_outputs[b].set_gradients(get_layer_index(), std::vector<double>(N_this, 0.0));
      batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), std::vector<double>());
    }
    return;
  }

  const int t_start = static_cast<int>(num_time_steps) - 1;
  int t_end = (bptt_max_ticks > 0) ? std::max(0, t_start - bptt_max_ticks + 1) : 0;

  auto run_hidden_gradients = [&](size_t start, size_t end)
  {
    // Total grad_matrix for each batch item in this chunk
    std::vector<std::vector<double>> chunk_grad_matrix(end - start, std::vector<double>(N_this, 0.0));
    // RNN grad matrix for each batch item in this chunk
    std::vector<std::vector<double>> chunk_rnn_grad_matrix(end - start, std::vector<double>(num_time_steps * N_this, 0.0));
    // d_next_h for each batch item in this chunk
    std::vector<std::vector<double>> chunk_d_next_h(end - start, std::vector<double>(N_this, 0.0));

    // Pre-calculate contribution from next layer for this chunk
    std::vector<std::vector<double>> chunk_grad_from_next_all_t(end - start, std::vector<double>(num_time_steps * N_this, 0.0));

    for (size_t i = 0; i < N_this; ++i)
    {
      for (size_t k = 0; k < N_next; ++k)
      {
        const double w_ik = next_layer.get_weight_value((unsigned)i, (unsigned)k);
        if (w_ik == 0.0) continue;
        for (size_t b_idx = 0; b_idx < (end - start); ++b_idx)
        {
          size_t b = start + b_idx;
          const auto& next_grad_matrix = batch_next_grad_matrix[b];
          const bool next_is_time_distributed = (next_grad_matrix.size() == N_next * num_time_steps);
          const bool next_is_last_only = (next_grad_matrix.size() == N_next);

          if (next_is_time_distributed)
          {
            for (int t = t_start; t >= t_end; --t)
            {
              chunk_grad_from_next_all_t[b_idx][t * N_this + i] += next_grad_matrix[t * N_next + k] * w_ik;
            }
          }
          else if (next_is_last_only)
          {
            chunk_grad_from_next_all_t[b_idx][t_start * N_this + i] += next_grad_matrix[k] * w_ik;
          }
        }
      }
    }

    for (int t = t_start; t >= t_end; --t)
    {
      // 2. Compute current step gradients
      for (size_t b_idx = 0; b_idx < (end - start); ++b_idx)
      {
        size_t b = start + b_idx;
        const auto& hidden_states = batch_hidden_states[b].at(get_layer_index());
        const size_t t_offset = t * N_this;

        for (size_t i = 0; i < N_this; ++i)
        {
          const double upstream = chunk_grad_from_next_all_t[b_idx][t_offset + i] + chunk_d_next_h[b_idx][i];
          const double preact = hidden_states[t].get_pre_activation_sum_at_neuron((unsigned)i);
          const double deriv = get_activation().activate_derivative(preact);
          double g = upstream * deriv;
          if (!std::isfinite(g)) g = 0.0;
          chunk_rnn_grad_matrix[b_idx][t_offset + i] = g;
          chunk_grad_matrix[b_idx][i] += g;
        }
      }

      // 3. Propagate through recurrent weights
      for (size_t b_idx = 0; b_idx < (end - start); ++b_idx) std::fill(chunk_d_next_h[b_idx].begin(), chunk_d_next_h[b_idx].end(), 0.0);

      for (size_t i = 0; i < N_this; ++i)
      {
        for (size_t j = 0; j < N_this; ++j)
        {
          const double rw_ij = get_recurrent_weight_value((unsigned)i, (unsigned)j);
          if (rw_ij == 0.0) continue;
          for (size_t b_idx = 0; b_idx < (end - start); ++b_idx)
          {
            chunk_d_next_h[b_idx][i] += chunk_rnn_grad_matrix[b_idx][t * N_this + j] * rw_ij;
          }
        }
      }
    }

    // Copy results back to batch vectors
    for (size_t b_idx = 0; b_idx < (end - start); ++b_idx)
    {
      size_t b = start + b_idx;
      batch_gradients_and_outputs[b].set_gradients(get_layer_index(), chunk_grad_matrix[b_idx]);
      batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), chunk_rnn_grad_matrix[b_idx]);
    }
  };

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 2;
  if (batch_size < num_threads * 2) num_threads = 1;

  if (num_threads <= 1)
  {
    run_hidden_gradients(0, batch_size);
  }
  else
  {
    std::vector<std::future<void>> futures;
    size_t chunk_size = batch_size / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t start = t * chunk_size;
      size_t end = (t == num_threads - 1) ? batch_size : start + chunk_size;
      futures.push_back(std::async(std::launch::async, run_hidden_gradients, start, end));
    }
    for (auto& f : futures) f.get();
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