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
  unsigned num_neurons_in_next_layer, 
  double weight_decay,
  LayerType layer_type, 
  const activation::method& activation_method,
  const OptimiserType& optimiser_type, 
  int residual_layer_number,
  double dropout_rate
  ) :
  _layer_index(layer_index),
  _number_input_neurons(num_neurons_in_previous_layer),
  _number_output_neurons(num_neurons_in_this_layer),
  _layer_type(layer_type),
  _optimiser_type(optimiser_type),
  _activation(activation_method),
  _residual_layer_number(residual_layer_number)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  if (num_neurons_in_this_layer == 0) 
  {
    Logger::panic("Error: Creating a layer with 0 neurons.");
  }
  if (layer_type != LayerType::Input && num_neurons_in_previous_layer == 0) 
  {
    Logger::warning("Warning: Non-input layer created with 0 inputs.");
  }

  // create the weights
  resize_weights(_activation, _number_input_neurons, _number_output_neurons, weight_decay);

  _neurons.reserve(_number_output_neurons);
  for (unsigned neuron_number = 0; neuron_number < _number_output_neurons; ++neuron_number)
  {
    auto neuron = Neuron(
      layer_type == LayerType::Input ? 0 : num_neurons_in_previous_layer,
      _number_output_neurons,
      layer_type == LayerType::Output ? 0 : num_neurons_in_next_layer,
      neuron_number, 
      dropout_rate == 0.0 ? Neuron::Type::Normal : Neuron::Type::Dropout,
      dropout_rate);
    _neurons.emplace_back(neuron);
  }
}

ElmanRNNLayer::ElmanRNNLayer(const ElmanRNNLayer& src) noexcept :
  _layer_index(src._layer_index),
  _neurons(src._neurons),
  _number_input_neurons(src._number_input_neurons),
  _number_output_neurons(src._number_output_neurons),
  _layer_type(src._layer_type),
  _weights(src._weights),
  _recurrent_weights(src._recurrent_weights),
  _bias_weights(src._bias_weights),
  _optimiser_type(src._optimiser_type),
  _activation(src._activation),
  _residual_layer_number(src._residual_layer_number)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
}

ElmanRNNLayer::ElmanRNNLayer(
  unsigned layer_index,
  const std::vector<Neuron>& neurons,
  unsigned number_input_neurons,
  LayerType layer_type,
  OptimiserType optimiser_type,
  int residual_layer_number,
  const activation::method& activation_method,
  const std::vector<std::vector<WeightParam>>& weights,
  const std::vector<std::vector<WeightParam>>& recurrent_weights,
  const std::vector<WeightParam>& bias_weights
) : 
  _layer_index(layer_index),
  _neurons(neurons),
  _number_input_neurons(number_input_neurons),
  _number_output_neurons( static_cast<unsigned>(neurons.size())),
  _layer_type(layer_type),
  _optimiser_type(optimiser_type),
  _activation(activation_method),
  _weights(weights),
  _recurrent_weights(recurrent_weights),
  _bias_weights(bias_weights),
  _residual_layer_number(residual_layer_number)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
}

ElmanRNNLayer::ElmanRNNLayer(ElmanRNNLayer&& src) noexcept :
  _layer_index(src._layer_index),
  _neurons(std::move(src._neurons)),
  _number_input_neurons(src._number_input_neurons),
  _number_output_neurons(src._number_output_neurons),
  _layer_type(src._layer_type),
  _weights(std::move(src._weights)),
  _recurrent_weights(std::move(src._recurrent_weights)),
  _bias_weights(std::move(src._bias_weights)),
  _optimiser_type(std::move(src._optimiser_type)),
  _activation(std::move(src._activation)),
  _residual_layer_number(src._residual_layer_number)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  src._layer_index = 0;
  src._number_output_neurons = 0;
  src._number_input_neurons = 0;
  src._optimiser_type = OptimiserType::None;
}

ElmanRNNLayer& ElmanRNNLayer::operator=(const ElmanRNNLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  if(this != &src)
  {
    _layer_index = src._layer_index;
    _neurons = src._neurons;
    _number_input_neurons = src._number_input_neurons;
    _number_output_neurons = src._number_output_neurons;
    _layer_type = src._layer_type;
    _weights = src._weights;
    _recurrent_weights = src._recurrent_weights;
    _bias_weights = src._bias_weights;
    _optimiser_type = src._optimiser_type;
    _activation = src._activation;
  }
  return *this;
}

ElmanRNNLayer& ElmanRNNLayer::operator=(ElmanRNNLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  if(this != &src)
  {
    _layer_index = src._layer_index;
    _neurons = std::move(src._neurons);
    _number_input_neurons = src._number_input_neurons;
    _number_output_neurons = src._number_output_neurons;
    _layer_type = src._layer_type;
    _weights = std::move(src._weights);
    _recurrent_weights = std::move(src._recurrent_weights);
    _bias_weights = std::move(src._bias_weights);
    _optimiser_type = std::move(src._optimiser_type);
    _activation = std::move(src._activation);
   
    src._number_output_neurons = 0;
    src._number_input_neurons = 0;
    src._optimiser_type = OptimiserType::None;
  }
  return *this;
}

ElmanRNNLayer::~ElmanRNNLayer() = default;

void ElmanRNNLayer::resize_weights(
  const activation& activation_method,
  unsigned number_input_neurons,
  unsigned number_output_neurons, 
  double weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  if (has_bias())
  {
    _bias_weights.reserve(number_output_neurons);
    auto weights = activation_method.weight_initialization(number_output_neurons, 1);
    for (unsigned o = 0; o < number_output_neurons; ++o)
    {
      const auto& weight = weights[o];
      _bias_weights.emplace_back(WeightParam(weight, 0.0, 0.0, 0.0, 0.0));
    }
  }

  if (number_input_neurons > 0)
  {
	  _weights.resize(number_input_neurons);
	  for (unsigned i = 0; i < number_input_neurons; ++i)
	  {
		auto weights = activation_method.weight_initialization(number_output_neurons, number_input_neurons);
		assert(weights.size() == number_output_neurons);
		_weights[i].reserve(number_output_neurons);
		for (unsigned o = 0; o < number_output_neurons; ++o)
		{
		  const auto& weight = weights[o];
		  _weights[i].emplace_back(WeightParam(weight, 0.0, 0.0, 0.0, weight_decay));
		}
	  }
  }

  _recurrent_weights.resize(number_output_neurons);
  for (unsigned i = 0; i < number_output_neurons; ++i)
  {
    auto weights = activation_method.weight_initialization(number_output_neurons, number_output_neurons);
    assert(weights.size() == number_output_neurons);
    _recurrent_weights[i].reserve(number_output_neurons);
    for (unsigned o = 0; o < number_output_neurons; ++o)
    {
      const auto& weight = weights[o];
      _recurrent_weights[i].emplace_back(WeightParam(weight, 0.0, 0.0, 0.0, weight_decay));
    }
  }
}

bool ElmanRNNLayer::has_bias() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  return _has_bias_neuron;
}

unsigned ElmanRNNLayer::number_neurons() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  return _number_output_neurons;
}

const std::vector<Neuron>& ElmanRNNLayer::get_neurons() const noexcept
{ 
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  return _neurons;
}

std::vector<Neuron>& ElmanRNNLayer::get_neurons() noexcept
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  return _neurons;
}

const Neuron& ElmanRNNLayer::get_neuron(unsigned index) const 
{ 
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  if (index >= _neurons.size()) 
  {
    throw std::out_of_range("Index out of bounds in ElmanRNNLayer::get_neuron.");
  }
  return _neurons[index];
}

Neuron& ElmanRNNLayer::get_neuron(unsigned index) 
{ 
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  if (index >= _neurons.size()) 
  {
    throw std::out_of_range("Index out of bounds in ElmanRNNLayer::get_neuron.");
  }
  return _neurons[index];
}

std::vector<double> ElmanRNNLayer::calculate_forward_feed(
  GradientsAndOutputs& gradients_and_outputs,
  const BaseLayer& previous_layer,
  const std::vector<double>& previous_layer_inputs,
  const std::vector<double>&, // residual_output_values is not used
  std::vector<HiddenState>& hidden_states,
  bool is_training) const
{
    const size_t N_prev = previous_layer.number_neurons();
    const size_t N_this = number_neurons();
    const size_t num_time_steps = N_prev > 0 ? previous_layer_inputs.size() / N_prev : 0;

    std::vector<double> output_sequence(num_time_steps * N_this, 0.0);
    std::vector<double> last_output_sequence(N_this, 0.0);
    
    // hidden_states is already sized for a single batch item, so no need to check empty()
    // and hidden_states.assign (it's passed by reference and managed by NeuralNetwork)
    assert(hidden_states.size() == num_time_steps);

    std::vector<double> prev_hidden_state_values(N_this, 0.0);

    for (size_t t = 0; t < num_time_steps; ++t) {
        std::vector<double> current_input_t(N_prev);
        for(size_t i = 0; i < N_prev; ++i) {
            current_input_t[i] = previous_layer_inputs[t * N_prev + i];
        }

        std::vector<double> pre_activation_sums(N_this, 0.0);

        for (size_t j = 0; j < N_this; ++j) {
            if (!_bias_weights.empty()) {
                pre_activation_sums[j] = _bias_weights[j].get_value();
            }
        }

        if (_layer_type != LayerType::Input) {
            for (size_t i = 0; i < N_prev; ++i) {
                for (size_t j = 0; j < N_this; ++j) {
                    pre_activation_sums[j] += current_input_t[i] * _weights[i][j].get_value();
                }
            }
        }
        
        // BUG: This block of code corrupts the pre-activation sums even when
        // prev_hidden_state_values is a zero vector. This prevents the network
        // from learning. It is commented out to allow the layer to function
        // as a feed-forward layer, but this breaks its recurrent functionality
        // for sequences with more than one time step.
        if (_layer_type == LayerType::Hidden || _layer_type == LayerType::Output) 
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
                pre_activation_sums[j] += h_prev_i * _recurrent_weights[i][j].get_value();
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
            // t == 0: initial hidden state is assumed zero — explicit log helps debugging
            Logger::trace([=] 
            {
              return Logger::factory("RNN Forward t=0: skipping recurrent add (initial hidden state = 0).");
             });
          }
        }

        std::vector<double> current_hidden_state_values(N_this);
        for (size_t j = 0; j < N_this; ++j) {
            const auto& neuron = get_neuron((unsigned)j);
            double output = get_activation().activate(pre_activation_sums[j]);

            if (is_training && neuron.is_dropout()) {
                if (neuron.must_randomly_drop()) {
                    output = 0.0;
                } else {
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
  switch (error_calculation_type)
  {
  case ErrorCalculation::type::mse:
    return calculate_mse_error_deltas(deltas, target_outputs, given_outputs);
  case ErrorCalculation::type::bce_loss:
    return calculate_bce_error_deltas(deltas, target_outputs, given_outputs);
  default:
    throw std::invalid_argument("ErrorCalculation type is not supported for ElmanRNNLayer!");
  }
}

void ElmanRNNLayer::calculate_bce_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs) const
{
  const size_t N_total = number_neurons();

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    deltas[neuron_index] = given_outputs[neuron_index] - target_outputs[neuron_index];
  }
}

void ElmanRNNLayer::calculate_mse_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs) const
{
  const size_t N_total = number_neurons();

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    deltas[neuron_index] = given_outputs[neuron_index] - target_outputs[neuron_index];
  }
}

void ElmanRNNLayer::calculate_output_gradients(
  GradientsAndOutputs& gradients_and_outputs,
  const std::vector<double>& target_outputs,
  const std::vector<HiddenState>& hidden_states,
  double gradient_clip_threshold,
  ErrorCalculation::type error_calculation_type) const
{
  const size_t N_total = number_neurons();

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

    for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
    {
      double deriv = 1.0;
      if (hs_ptr != nullptr)
      {
        deriv = get_activation().activate_derivative(hs_ptr->get_pre_activation_sum_at_neuron(neuron_index));
      }
      double g = deltas[neuron_index] * deriv;
      gradients[neuron_index] = clip_gradient(g, gradient_clip_threshold);
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

    // Compute gradient for each neuron using last time-step output and its pre-activation derivative
    for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
    {
      const size_t last_idx = (num_time_steps - 1) * N_total + neuron_index;
      const double last_output = given_outputs[last_idx];
      const double target = (neuron_index < target_outputs.size()) ? target_outputs[neuron_index] : 0.0;
      const double delta = last_output - target;

      double deriv = 1.0;
      if (last_hs_ptr != nullptr)
      {
        deriv = get_activation().activate_derivative(last_hs_ptr->get_pre_activation_sum_at_neuron(neuron_index));
      }

      double g = delta * deriv;
      gradients[neuron_index] = clip_gradient(g, gradient_clip_threshold);
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
  const BaseLayer& next_layer,
  const std::vector<double>& next_grad_matrix,
  const std::vector<double>&,
  const std::vector<HiddenState>& hidden_states,
  double gradient_clip_threshold) const
{
  const size_t num_time_steps = hidden_states.size();
  const size_t N_this = number_neurons();
  const size_t N_next = next_layer.number_neurons();

  // Defensive: no time steps => nothing to do
  if (num_time_steps == 0 || N_this == 0)
  {
    gradients_and_outputs.set_gradients(get_layer_index(), std::vector<double>(N_this, 0.0));
    gradients_and_outputs.set_rnn_gradients(get_layer_index(), std::vector<double>());
    Logger::trace([=] { 
      return Logger::factory("HiddenGradients: empty sequence (num_time_steps=", num_time_steps, ", N_this=", N_this, ")"); 
    });
    return;
  }

  Logger::trace([=] {
    return Logger::factory(
      "HiddenGradients START: num_time_steps=", num_time_steps,
      ", N_this=", N_this,
      ", N_next=", N_next,
      ", next_grad_matrix_size=", next_grad_matrix.size());
    });

  std::vector<double> grad_matrix(N_this, 0.0);                     // aggregated dE/dz over time
  std::vector<double> rnn_grad_matrix(num_time_steps * N_this, 0.0); // per-time dE/dz
  std::vector<double> d_next_h(N_this, 0.0);                        // dE/dh(t) from t+1

  // Backprop through time: iterate from last timestep to first.
  const int t_start = static_cast<int>(num_time_steps) - 1;
  for (int t = t_start; t >= 0; --t)
  {
    // grad_from_next_layer represents contribution from the next (non-recurrent) layer.
    // This is only non-zero for the final time-step when the next layer consumes
    // the last hidden-state (common for many setups). Ensure the caller provides
    // next_grad_matrix accordingly.
    std::vector<double> grad_from_next_layer(N_this, 0.0);
    if (t == t_start)
    {
      // Defensive: next_grad_matrix should be of size N_next
      assert(next_grad_matrix.size() == N_next || N_next == 0);
      for (size_t k = 0; k < N_next; ++k)
      {
        for (size_t i = 0; i < N_this; ++i)
        {
          // next_layer.get_weight_param(input_index, neuron_index)
          grad_from_next_layer[i] += next_grad_matrix[k] * next_layer.get_weight_param(i, k).get_value();
        }
      }

      // log a few values (or none if empty)
      Logger::trace([=] {
        return Logger::factory(
          "HiddenGradients t=", t, " grad_from_next_layer[0..3]=",
          (grad_from_next_layer.size() > 0 ? grad_from_next_layer[0] : 0.0), ",",
          (grad_from_next_layer.size() > 1 ? grad_from_next_layer[1] : 0.0), ",",
          (grad_from_next_layer.size() > 2 ? grad_from_next_layer[2] : 0.0), ",",
          (grad_from_next_layer.size() > 3 ? grad_from_next_layer[3] : 0.0));
        });
    }

    // Compute gradients for this time-step
    std::vector<double> grad_matrix_t(N_this, 0.0);
    for (size_t i = 0; i < N_this; ++i)
    {
      // total gradient into pre-activation z_i(t)
      double grad_from_layer_and_time = grad_from_next_layer[i] + d_next_h[i];

      // derivative uses the stored pre-activation sums for this timestep
      double deriv = get_activation().activate_derivative(hidden_states[static_cast<size_t>(t)].get_pre_activation_sum_at_neuron((unsigned)i));
      double g = grad_from_layer_and_time * deriv;
      grad_matrix_t[i] = clip_gradient(g, gradient_clip_threshold);

      // accumulate across time for final gradients used to update recurrent biases/other params
      grad_matrix[i] += grad_matrix_t[i];

      // store per-time dE/dz for use when computing weight gradients
      rnn_grad_matrix[static_cast<size_t>(t) * N_this + i] = grad_matrix_t[i];
    }

    Logger::trace([=] 
      {
        // log summary for this timestep
        double sum_abs = 0.0;
        double max_abs = 0.0;
        for (auto v : grad_matrix_t) {
          sum_abs += std::fabs(v); max_abs = std::max(max_abs, std::fabs(v));
        }
        return Logger::factory("HiddenGradients t=", t, " grad_matrix_t_sumabs=", sum_abs, ", maxabs=", max_abs);
      });

    // Compute effect on previous hidden state h(t-1):
    // d_next_h[i_prev] = sum_j grad_matrix_t[j] * W_h(prev=i_prev -> curr=j)
    // iterate j (current neuron) then accumulate for each prev neuron i_prev
    std::fill(d_next_h.begin(), d_next_h.end(), 0.0);
    for (size_t j = 0; j < N_this; ++j)
    {
        const double g_j = grad_matrix_t[j];
        if (g_j == 0.0) continue;
        for (size_t i = 0; i < N_this; ++i)
        {
            d_next_h[i] += g_j * _recurrent_weights[i][j].get_value();
        }
    }    
    
    Logger::trace([=] 
    {
      double sum_abs = 0.0;
      double max_abs = 0.0;
      for (auto v : d_next_h) 
      { 
        sum_abs += std::fabs(v); max_abs = std::max(max_abs, std::fabs(v)); 
      }
      return Logger::factory("HiddenGradients after t=", t, " d_next_h_sumabs=", sum_abs, ", maxabs=", max_abs);
    });
  }

  // final aggregated gradient stats
    Logger::trace([=] 
    {
      double norm = 0.0;
      double max_abs = 0.0;
      for (auto v : grad_matrix) { 
        norm += v * v; max_abs = std::max(max_abs, std::fabs(v)); 
      }
      norm = std::sqrt(norm);
      return Logger::factory("HiddenGradients END: grad_matrix_norm=", norm, ", grad_matrix_maxabs=", max_abs,
        ", sample_grad_matrix[0..3]=",
        (grad_matrix.size() > 0 ? grad_matrix[0] : 0.0), ",",
        (grad_matrix.size() > 1 ? grad_matrix[1] : 0.0), ",",
        (grad_matrix.size() > 2 ? grad_matrix[2] : 0.0), ",",
        (grad_matrix.size() > 3 ? grad_matrix[3] : 0.0));
    });

    // also log first/last entries of rnn_grad_matrix to check time distribution
    Logger::trace([=] {
      return Logger::factory("HiddenGradients rnn_grad_matrix first,last =",
        rnn_grad_matrix.front(),
        ",",
        rnn_grad_matrix.back());
    });

  gradients_and_outputs.set_gradients(get_layer_index(), grad_matrix);
  gradients_and_outputs.set_rnn_gradients(get_layer_index(), rnn_grad_matrix);
}

void ElmanRNNLayer::apply_weight_gradient(const double gradient, const double learning_rate, bool is_bias, WeightParam& weight_param, double clipping_scale, double gradient_clip_threshold)
{
  auto clipped_gradient = clipping_scale <= 0.0 ? clip_gradient(gradient, gradient_clip_threshold) : gradient * clipping_scale;
  
  double final_gradient = clipped_gradient;
  if (!is_bias && weight_param.get_weight_decay() > 0.0)
  {
    final_gradient += weight_param.get_weight_decay() * weight_param.get_value();
  }

  double new_weight = weight_param.get_value() - learning_rate * final_gradient;
  weight_param.set_raw_gradient(clipped_gradient);
  weight_param.set_value(new_weight);
}

double ElmanRNNLayer::clip_gradient(double gradient, double gradient_clip_threshold)
{
  if (!std::isfinite(gradient))
  {
    return 0.0;
  }

  if (gradient > gradient_clip_threshold)
  {
    return gradient_clip_threshold;
  }
  if (gradient < -gradient_clip_threshold)
  {
    return -gradient_clip_threshold;
  }
  return gradient;
}

unsigned int ElmanRNNLayer::get_layer_index() const noexcept
{
    return _layer_index;
}

BaseLayer::LayerType ElmanRNNLayer::layer_type() const
{
    return _layer_type;
}

unsigned int ElmanRNNLayer::number_input_neurons(bool add_bias) const noexcept
{
    return _number_input_neurons + (add_bias ? 1 : 0);
}

const std::vector<std::vector<WeightParam>>& ElmanRNNLayer::get_weight_params() const
{
    return _weights;
}

const WeightParam& ElmanRNNLayer::get_weight_param(unsigned int input_neuron_number, unsigned int neuron_index) const
{
    return _weights[input_neuron_number][neuron_index];
}

WeightParam& ElmanRNNLayer::get_weight_param(unsigned int input_neuron_number, unsigned int neuron_index)
{
    return _weights[input_neuron_number][neuron_index];
}

const std::vector<WeightParam>& ElmanRNNLayer::get_bias_weight_params() const
{
    return _bias_weights;
}

const activation& ElmanRNNLayer::get_activation() const noexcept
{
    return _activation;
}

WeightParam& ElmanRNNLayer::get_bias_weight_param(unsigned int neuron_index)
{
    return _bias_weights[neuron_index];
}

unsigned int ElmanRNNLayer::get_number_output_neurons() const
{
    return _number_output_neurons;
}

const OptimiserType ElmanRNNLayer::get_optimiser_type() const noexcept
{
    return _optimiser_type;
}

const std::vector<std::vector<WeightParam>>& ElmanRNNLayer::get_recurrent_weight_params() const
{
    return _recurrent_weights;
}

std::vector<std::vector<WeightParam>>& ElmanRNNLayer::get_recurrent_weight_params()
{
    return _recurrent_weights;
}

BaseLayer* ElmanRNNLayer::clone() const
{
    return new ElmanRNNLayer(*this);
}

int ElmanRNNLayer::residual_layer_number() const
{
    return _residual_layer_number;
}

const std::vector<std::vector<WeightParam>>& ElmanRNNLayer::get_residual_weight_params() const
{
    return _residual_weights;
}

std::vector<std::vector<WeightParam>>& ElmanRNNLayer::get_residual_weight_params()
{
    return _residual_weights;
}

std::vector<WeightParam>& ElmanRNNLayer::get_residual_weight_params(unsigned int neuron_index)
{
    return _residual_weights[neuron_index];
}