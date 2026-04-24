#pragma once

#include "./libraries/instrumentor.h"

#include "activation.h"
#include "errorcalculation.h"
#include "evaluationconfig.h"
#include "gradientsandoutputs.h"
#include "hiddenstates.h"
#include "neuralnetworkhelpermetrics.h"
#include "neuron.h"
#include "optimiser.h"
#include "residualprojector.h"
#include "taskqueue.h"
#include "weightparam.h"

#include <cmath>
#include <memory>
#include <span>
#include <vector>

class layer_activation_helper
{
public:
  struct range
  {
    unsigned start;
    unsigned end; // Exclusive
    activation activation_method;

    range(unsigned s, unsigned e, const activation& m) : 
      start(s), 
      end(e), 
      activation_method(m) 
    {
      MYODDWEB_PROFILE_FUNCTION("layer_activation_helper::range");
    }
  };

public:
  layer_activation_helper(const activation& activation_method, unsigned number_input_neurons, unsigned number_output_neurons) noexcept :
    _number_input_neurons(number_input_neurons),
    _number_output_neurons(number_output_neurons)
  {
    MYODDWEB_PROFILE_FUNCTION("layer_activation_helper");
    _ranges.emplace_back(0, number_output_neurons, activation_method);
  }

  ~layer_activation_helper()
  {
    MYODDWEB_PROFILE_FUNCTION("layer_activation_helper");
  }

  layer_activation_helper(const layer_activation_helper& src) : 
    _ranges(src._ranges),
    _number_input_neurons(src._number_input_neurons),
    _number_output_neurons(src._number_output_neurons)
  {
    MYODDWEB_PROFILE_FUNCTION("layer_activation_helper");
  }

  layer_activation_helper(layer_activation_helper&& src) noexcept :
    _ranges(std::move(src._ranges)),
    _number_input_neurons(src._number_input_neurons),
    _number_output_neurons(src._number_output_neurons)
  {
    MYODDWEB_PROFILE_FUNCTION("layer_activation_helper");
  }

  layer_activation_helper& operator=(const layer_activation_helper& src)
  {
    MYODDWEB_PROFILE_FUNCTION("layer_activation_helper");
    if (this != &src)
    {
      _ranges = src._ranges;
      _number_input_neurons = src._number_input_neurons;
      _number_output_neurons = src._number_output_neurons;
    }
    return *this;
  }

  layer_activation_helper& operator=(layer_activation_helper&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("layer_activation_helper");
    if (this != &src)
    {
      _ranges = std::move(src._ranges);
      _number_input_neurons = src._number_input_neurons;
      _number_output_neurons = src._number_output_neurons;
    }
    return *this;
  }

  void set_bounds(const activation& activation_method, unsigned start, unsigned end)
  {
    MYODDWEB_PROFILE_FUNCTION("layer_activation_helper");
#if VALIDATE_DATA == 1
    if (end <= start)
    {
      Logger::panic("Trying to set a range of neurons ... but the start is past the end!");
    }
    if (end > _number_output_neurons)
    {
      Logger::panic("Trying to set a neuron:, ", end, " past the number of available neurons!");
    }
#endif

    // Remove or split existing ranges that overlap with [start, end)
    std::vector<range> new_ranges;
    for (const auto& r : _ranges)
    {
      if (r.end <= start || r.start >= end)
      {
        // No overlap
        new_ranges.push_back(r);
      }
      else
      {
        // Overlap - add parts outside [start, end)
        if (r.start < start)
        {
          new_ranges.emplace_back(r.start, start, r.activation_method);
        }
        if (r.end > end)
        {
          new_ranges.emplace_back(end, r.end, r.activation_method);
        }
      }
    }
    new_ranges.emplace_back(start, end, activation_method);
    
    // Keep it sorted
    std::sort(new_ranges.begin(), new_ranges.end(), [](const range& a, const range& b) {
      return a.start < b.start;
    });
    _ranges = std::move(new_ranges);
  }

  [[nodiscard]] inline const activation& get_activation(unsigned output_neuron_number) const
  {
    MYODDWEB_PROFILE_FUNCTION("layer_activation_helper");
    // Usually only 1 range, so linear search is fine.
    for (const auto& r : _ranges)
    {
      if (output_neuron_number >= r.start && output_neuron_number < r.end)
      {
        return r.activation_method;
      }
    }
    Logger::panic("Trying to get an activation method for neuron ", output_neuron_number, " which is not covered by any range!");
    static activation dummy(activation::method::linear, 0.0);
    return dummy;
  }

  [[nodiscard]] inline const std::vector<range>& ranges() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("layer_activation_helper");
    return _ranges;
  }

  [[nodiscard]] inline unsigned get_number_input_neurons() const
  {
    MYODDWEB_PROFILE_FUNCTION("layer_activation_helper");
    return _number_input_neurons;
  }

  [[nodiscard]] inline unsigned get_number_output_neurons() const
  {
    MYODDWEB_PROFILE_FUNCTION("layer_activation_helper");
    return _number_output_neurons;
  }

  [[nodiscard]] inline double weight_initialization(unsigned output_neuron_number) const
  {
    MYODDWEB_PROFILE_FUNCTION("layer_activation_helper");
    for (const auto& r : _ranges)
    {
      if (output_neuron_number >= r.start && output_neuron_number < r.end)
      {
        return r.activation_method.weight_initialization(get_number_input_neurons(), r.end - r.start);
      }
    }
    Logger::panic("Trying to initialize weight for neuron ", output_neuron_number, " which is not covered by any range!");
  }

  inline void scale_temperature(double factor) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("layer_activation_helper");
    for (auto& r : _ranges)
    {
      r.activation_method.scale_temperature(factor);
    }
  }

  [[nodiscard]] inline double get_temperature(unsigned range_index) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("layer_activation_helper");
#if VALIDATE_DATA == 1
    if (range_index >= _ranges.size())
    {
      Logger::panic("Trying to get temperature for range ", range_index, " which is out of bounds!");
    }
#endif
    return _ranges[range_index].activation_method.get_temperature();
  }

  inline void scale_temperature(unsigned range_index, double factor) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("layer_activation_helper");
#if VALIDATE_DATA == 1
    if (range_index >= _ranges.size())
    {
      Logger::panic("Trying to scale temperature for range ", range_index, " which is out of bounds!");
    }
#endif
    _ranges[range_index].activation_method.scale_temperature(factor);
  }

private:
  std::vector<range> _ranges;
  unsigned _number_input_neurons;
  unsigned _number_output_neurons;
};

class Layer
{
public:
  enum class Architecture
  {
    None,
    FF,
    Elman,
    Gru,
    Lstm,
    MultiOutput
  };

  enum class Role
  {
    Input,
    Hidden,
    Output,
    MultiOutput
  };

  [[nodiscard]] inline static std::string architecture_to_string(const Architecture& architecture)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    switch (architecture)
    {
    case Architecture::None:
      return "None";

    case Architecture::FF:
      return "FF";

    case Architecture::Elman:
      return "Elman";

    case Architecture::Gru:
      return "Gru";

    case Architecture::Lstm:
      return "Lstm";

    case Architecture::MultiOutput:
      return "MultiOutput";

    default:
      Logger::panic("Unknown Layer architecture: ", (int)architecture);
    }
  }

  [[nodiscard]] inline static Architecture architecture_from_string(const std::string& str)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    std::string lower_str = str;
    // Convert the string to lowercase for case-insensitive comparison
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
      [](unsigned char c) { return std::tolower(c); });

    if (lower_str == "none")
    {
      return Architecture::None;
    }
    if (lower_str == "ff")
    {
      return Architecture::FF;
    }
    if (lower_str == "elman")
    {
      return Architecture::Elman;
    }
    if (lower_str == "gru")
    {
      return Architecture::Gru;
    }
    if (lower_str == "lstm")
    {
      return Architecture::Lstm;
    }
    if (lower_str == "multioutput")
    {
      return Architecture::MultiOutput;
    }
    Logger::panic("Unknown Layer architecture: ", str);
  }

  [[nodiscard]] static std::vector<double> create_w_decays(unsigned number_input_neurons, unsigned number_output_neurons, double decay)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return std::vector<double>(static_cast<size_t>(number_input_neurons) * number_output_neurons, decay);
  }

  Layer(const Layer& src) noexcept :
    _layer_index(src._layer_index),
    _layer_role(src._layer_role),
    _optimiser_type(src._optimiser_type),
    _residual_layer_number(src._residual_layer_number),
    _neurons(src._neurons),
    _w_values(src._w_values),
    _w_grads(src._w_grads),
    _w_velocities(src._w_velocities),
    _w_m1(src._w_m1),
    _w_m2(src._w_m2),
    _w_timesteps(src._w_timesteps),
    _w_decays(src._w_decays),
    _b_values(src._b_values),
    _b_grads(src._b_grads),
    _b_velocities(src._b_velocities),
    _b_m1(src._b_m1),
    _b_m2(src._b_m2),
    _b_timesteps(src._b_timesteps),
    _b_decays(src._b_decays),
    _residual_projector(nullptr),
    _task_queue_pool(nullptr),
    _layer_activation_helper(src._layer_activation_helper),
    _momentum(src._momentum)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (src._residual_projector != nullptr)
    {
      _residual_projector = new ResidualProjector(*src._residual_projector);
    }
    if (src._task_queue_pool)
    {
      _task_queue_pool = std::make_unique<TaskQueuePool<void>>(src._task_queue_pool->get_number_of_threads());
    }
  }

  Layer(Layer&& src) noexcept :
    _layer_index(src._layer_index),
    _layer_role(src._layer_role),
    _optimiser_type(std::move(src._optimiser_type)),
    _neurons(std::move(src._neurons)),
    _w_values(std::move(src._w_values)),
    _w_grads(std::move(src._w_grads)),
    _w_velocities(std::move(src._w_velocities)),
    _w_m1(std::move(src._w_m1)),
    _w_m2(std::move(src._w_m2)),
    _w_timesteps(std::move(src._w_timesteps)),
    _w_decays(std::move(src._w_decays)),
    _b_values(std::move(src._b_values)),
    _b_grads(std::move(src._b_grads)),
    _b_velocities(std::move(src._b_velocities)),
    _b_m1(std::move(src._b_m1)),
    _b_m2(std::move(src._b_m2)),
    _b_timesteps(std::move(src._b_timesteps)),
    _b_decays(std::move(src._b_decays)),
    _residual_layer_number(src._residual_layer_number),
    _residual_projector(src._residual_projector),
    _task_queue_pool(std::move(src._task_queue_pool)),
    _layer_activation_helper(std::move(src._layer_activation_helper)),
    _momentum(src._momentum)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    src._layer_role = Role::Input;
    src._layer_index = 0;
    src._optimiser_type = OptimiserType::None;
    src._residual_layer_number = 0;
    src._residual_projector = nullptr;
  }

  Layer& operator=(const Layer& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (this != &src)
    {
      _layer_index = src._layer_index;
      _layer_role = src._layer_role;
      _optimiser_type = src._optimiser_type;
      _neurons = src._neurons;

      _w_values = src._w_values;
      _w_grads = src._w_grads;
      _w_velocities = src._w_velocities;
      _w_m1 = src._w_m1;
      _w_m2 = src._w_m2;
      _w_timesteps = src._w_timesteps;
      _w_decays = src._w_decays;

      _b_values = src._b_values;
      _b_grads = src._b_grads;
      _b_velocities = src._b_velocities;
      _b_m1 = src._b_m1;
      _b_m2 = src._b_m2;
      _b_timesteps = src._b_timesteps;
      _b_decays = src._b_decays;

      _residual_layer_number = src._residual_layer_number;
      delete _residual_projector;
      _residual_projector = nullptr;
      if (src._residual_projector != nullptr)
      {
        _residual_projector = new ResidualProjector(*src._residual_projector);
      }

      if (src._task_queue_pool)
      {
        _task_queue_pool = std::make_unique<TaskQueuePool<void>>(src._task_queue_pool->get_number_of_threads());
      }
      else
      {
        _task_queue_pool.reset();
      }
      _layer_activation_helper = src._layer_activation_helper;
      _momentum = src._momentum;
    }
    return *this;
  }

  Layer& operator=(Layer&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (this != &src)
    {
      _layer_index = src._layer_index;
      _layer_role = src._layer_role;
      _optimiser_type = std::move(src._optimiser_type);
      _neurons = std::move(src._neurons);

      _w_values = std::move(src._w_values);
      _w_grads = std::move(src._w_grads);
      _w_velocities = std::move(src._w_velocities);
      _w_m1 = std::move(src._w_m1);
      _w_m2 = std::move(src._w_m2);
      _w_timesteps = std::move(src._w_timesteps);
      _w_decays = std::move(src._w_decays);

      _b_values = std::move(src._b_values);
      _b_grads = std::move(src._b_grads);
      _b_velocities = std::move(src._b_velocities);
      _b_m1 = std::move(src._b_m1);
      _b_m2 = std::move(src._b_m2);
      _b_timesteps = std::move(src._b_timesteps);
      _b_decays = std::move(src._b_decays);

      delete _residual_projector;
      _residual_projector = src._residual_projector;

      _task_queue_pool = std::move(src._task_queue_pool);

      _layer_activation_helper = std::move(src._layer_activation_helper);

      src._layer_index = 0;
      src._optimiser_type = OptimiserType::None;
      src._residual_projector = nullptr;
    }
    return *this;
  }

  virtual ~Layer()
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    delete _residual_projector;
    _residual_projector = nullptr;
  }

  [[nodiscard]] virtual std::vector<std::vector<NeuralNetworkHelperMetrics>> calculate_output_metrics(
    const std::vector<ErrorCalculation::type>& error_types,
    const std::vector<std::vector<double>>& checking_outputs,
    const std::vector<std::vector<double>>& predictions
  ) const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    Logger::panic("Only output layers can calculate output metrics!");
  }

  [[nodiscard]] virtual Architecture get_layer_architecture() const = 0;

  [[nodiscard]] inline OptimiserType get_optimiser_type() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _optimiser_type;
  }

  [[nodiscard]] inline unsigned get_layer_index() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_index;
  }

  [[nodiscard]] inline Role get_layer_role() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_role;
  }

  [[nodiscard]] inline int get_residual_layer_number() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _residual_layer_number;
  }

  [[nodiscard]] const ResidualProjector* get_residual_projector() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _residual_projector;
  }

  [[nodiscard]] inline ResidualProjector* get_residual_projector() noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _residual_projector;
  }

  [[nodiscard]] inline unsigned get_number_neurons() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return get_number_output_neurons();
  }

  [[nodiscard]] inline const layer_activation_helper& get_activation_helper() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_activation_helper;
  }

  [[nodiscard]] inline layer_activation_helper& get_activation_helper()
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_activation_helper;
  }

  [[nodiscard]] inline unsigned get_number_input_neurons() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_activation_helper.get_number_input_neurons();
  }

  [[nodiscard]] inline unsigned get_number_output_neurons() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_activation_helper.get_number_output_neurons();
  }

  [[nodiscard]] virtual bool use_bptt() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return false;
  }

  virtual void calculate_forward_feed(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& previous_layer,
    const std::vector<std::vector<double>>& batch_residual_output_values,
    std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    bool is_training) const = 0;

  virtual void calculate_output_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    std::vector<std::vector<double>>::const_iterator target_outputs_begin,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size) const = 0;

  void calculate_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    ErrorCalculation::type error_calculation_type,
    const EvaluationConfig& evaluation_config,
    const activation::method activation_method,
    unsigned start_neuron,
    unsigned end_neuron) const;

  [[nodiscard]] inline const activation& get_activation() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
#if VALIDATE_DATA == 1
    if (_layer_role == LayerRole::Output)
    {
      Logger::panic("The output layer MUST pass the neuron number!");
    }
#endif
    return get_activation(0);
  }

  [[nodiscard]] inline const activation& get_activation(unsigned neuron_number ) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_activation_helper.get_activation(neuron_number);
  }

  [[nodiscard]] inline double get_dropout() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (get_number_neurons() == 0)
    {
      return 0.0;
    }
    return get_neuron(0).get_dropout_rate();
  }

  virtual void calculate_hidden_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& next_layer,
    const std::vector<std::vector<double>>& batch_next_grad_matrix,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    int bptt_max_ticks) const = 0;

  virtual void calculate_and_store_gradients(
    const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& hidden_states,
    const Layer& previous_layer,
    size_t batch_size,
    int bptt_max_ticks) = 0;

  virtual double get_gradient_norm_sq() const = 0;

  virtual void zero_gradients() { MYODDWEB_PROFILE_FUNCTION("Layer"); }

  virtual void apply_stored_gradients(double learning_rate, double clipping_scale) = 0;

  virtual void scale_temperature(double factor) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _layer_activation_helper.scale_temperature(factor);
  }

  [[nodiscard]] virtual double get_temperature(unsigned range_index) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_activation_helper.get_temperature(range_index);
  }

  virtual void scale_temperature(unsigned range_index, double factor) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _layer_activation_helper.scale_temperature(range_index, factor);
  }

  void apply_update_to_vector(
    std::vector<double>& values,
    std::vector<double>& grads,
    std::vector<double>& velocities,
    std::vector<double>& m1,
    std::vector<double>& m2,
    std::vector<long long>& timesteps,
    const std::vector<double>& decays,
    double learning_rate,
    double clipping_scale,
    bool is_bias,
    OptimiserType optimiser_type)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    const size_t n = values.size();
    if (n == 0)
    {
      return;
    }

    switch (optimiser_type)
    {
    case OptimiserType::None:
      for (size_t i = 0; i < n; ++i)
      {
        double grad = grads[i] * clipping_scale;
        values[i] -= learning_rate * grad;
        grads[i] = grad;
      }
      break;

    case OptimiserType::SGD:
    {
      const auto&  momentum = get_momentum();
      for (size_t i = 0; i < n; ++i)
      {
        double grad = grads[i] * clipping_scale;
        if (!is_bias && i < decays.size() && decays[i] > 0.0)
        {
          grad += decays[i] * values[i];
        }
        velocities[i] = momentum * velocities[i] + grad;
        values[i] -= learning_rate * velocities[i];
        grads[i] = grad;
      }
    }
    break;

    case OptimiserType::Adam:
    case OptimiserType::AdamW:
    {
      const double beta1 = get_momentum();
      const double beta2 = 0.999;
      const double epsilon = 1e-8;

      for (size_t i = 0; i < n; ++i)
      {
        double grad = grads[i] * clipping_scale;
        const long long t = ++timesteps[i];

        const double p1 = 1.0 - std::pow(beta1, t);
        const double p2 = 1.0 - std::pow(beta2, t);

        m1[i] = beta1 * m1[i] + (1.0 - beta1) * grad;
        m2[i] = beta2 * m2[i] + (1.0 - beta2) * (grad * grad);

        double m_hat = m1[i] / p1;
        double v_hat = m2[i] / p2;
        double update_step = m_hat / (std::sqrt(v_hat) + epsilon);

        double current_weight = values[i];
        if (optimiser_type == OptimiserType::AdamW && !is_bias && i < decays.size())
        {
          current_weight *= (1.0 - learning_rate * decays[i]);
        }
        values[i] = current_weight - learning_rate * update_step;
        grads[i] = grad;
      }
    }
    break;

    case OptimiserType::Nadam:
    case OptimiserType::NadamW:
    {
      const double beta1 = get_momentum();
      const double beta2 = 0.999;
      const double epsilon = 1e-8;

      for (size_t i = 0; i < n; ++i)
      {
        double grad = grads[i] * clipping_scale;
        const long long t = ++timesteps[i];

        const double p1 = 1.0 - std::pow(beta1, t);
        const double p2 = 1.0 - std::pow(beta2, t);

        m1[i] = beta1 * m1[i] + (1.0 - beta1) * grad;
        m2[i] = beta2 * m2[i] + (1.0 - beta2) * (grad * grad);

        double m_hat = m1[i] / p1;
        double v_hat = m2[i] / p2;

        double m_nadam = beta1 * m_hat + ((1.0 - beta1) * grad) / p1;
        double update_step = m_nadam / (std::sqrt(v_hat) + epsilon);

        double current_weight = values[i];
        if (optimiser_type == OptimiserType::NadamW && !is_bias && i < decays.size())
        {
          current_weight *= (1.0 - learning_rate * decays[i]);
        }
        values[i] = current_weight - learning_rate * update_step;
        grads[i] = grad;
      }
    }
    break;

    default:
      Logger::panic("Unknown optimizer type:", (int)optimiser_type);
    }
  }

  void apply_update_to_weight(
    std::vector<double>& values,
    std::vector<double>& grads,
    std::vector<double>& velocities,
    std::vector<double>& m1,
    std::vector<double>& m2,
    std::vector<long long>& timesteps,
    const std::vector<double>& decays,
    unsigned idx,
    double gradient,
    double learning_rate,
    double clipping_scale,
    OptimiserType optimiser_type,
    unsigned neuron_number)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");

    // validation
    if (!std::isfinite(gradient))
    {
      Logger::panic("Error while calculating input weigh gradient it invalid.");
    }

    if (clipping_scale < 0.0)
    {
      // If clipping scale is negative, we clip the gradient to a fixed range
      Logger::warning("Clipping gradient to a fixed range.");
    }

    double final_gradient = gradient * clipping_scale;

    // Detect gradient explosions before they impact weights
    if (std::abs(final_gradient) > 1e6)
    {
      Logger::panic("CRITICAL: Gradient too large! Grad: ", final_gradient, " lr: ", learning_rate);
    }
    else if (!std::isfinite(final_gradient))
    {
      Logger::panic("CRITICAL: Non-finite gradient detected! Grad: ", final_gradient);
    }

    // Log trace for some updates to avoid flooding
    if (idx == 0 && (timesteps.empty() || timesteps[idx] % 50 == 0))
    {
      Logger::trace([&]()
        {
      std::ostringstream ss;
      ss << "[Layer::apply_update_to_weight] layer=" << _layer_index
        << ", idx=" << idx
        << ", grad=" << gradient
        << ", final_grad=" << final_gradient
        << ", lr=" << learning_rate
        << ", val_before=" << values[idx];
          return ss.str();
        });
    }

    switch (optimiser_type)
    {
    case OptimiserType::None:
      values[idx] -= learning_rate * final_gradient;
      grads[idx] = final_gradient;
      break;

    case OptimiserType::SGD:
    {
      double grad = final_gradient;
      if (!is_bias_index(values) && decays.size() > idx && decays[idx] > 0.0)
      {
        grad += decays[idx] * values[idx];
      }
      double previous_velocity = velocities[idx];
      double velocity = get_momentum(neuron_number) * previous_velocity + grad;
      values[idx] -= learning_rate * velocity;
      velocities[idx] = velocity;
      grads[idx] = grad;
    }
    break;

    case OptimiserType::Adam:
    case OptimiserType::AdamW:
    {
      const double beta1 = get_momentum(neuron_number);
      const double beta2 = 0.999;
      const double epsilon = 1e-8;

      const long long time_step = ++timesteps[idx];

      m1[idx] = beta1 * m1[idx] + (1.0 - beta1) * final_gradient;
      m2[idx] = beta2 * m2[idx] + (1.0 - beta2) * (final_gradient * final_gradient);

      double m_hat = m1[idx] / (1.0 - std::pow(beta1, time_step));
      double v_hat = m2[idx] / (1.0 - std::pow(beta2, time_step));

      double update_step = m_hat / (std::sqrt(v_hat) + epsilon);

      double current_weight = values[idx];
      if (optimiser_type == OptimiserType::AdamW && !is_bias_index(values) && decays.size() > idx)
      {
        current_weight *= (1.0 - learning_rate * decays[idx]);
      }

      values[idx] = current_weight - learning_rate * update_step;
      grads[idx] = final_gradient;
    }
    break;

    case OptimiserType::Nadam:
    case OptimiserType::NadamW:
    {
      const double beta1 = get_momentum(neuron_number);
      const double beta2 = 0.999;
      const double epsilon = 1e-8;

      const long long time_step = ++timesteps[idx];

      m1[idx] = beta1 * m1[idx] + (1.0 - beta1) * final_gradient;
      m2[idx] = beta2 * m2[idx] + (1.0 - beta2) * (final_gradient * final_gradient);

      double m_hat = m1[idx] / (1.0 - std::pow(beta1, time_step));
      double v_hat = m2[idx] / (1.0 - std::pow(beta2, time_step));

      double m_nadam = beta1 * m_hat + ((1.0 - beta1) * final_gradient) / (1.0 - std::pow(beta1, time_step));
      double update_step = m_nadam / (std::sqrt(v_hat) + epsilon);

      double current_weight = values[idx];
      if (optimiser_type == OptimiserType::NadamW && !is_bias_index(values) && decays.size() > idx)
      {
        current_weight *= (1.0 - learning_rate * decays[idx]);
      }

      values[idx] = current_weight - learning_rate * update_step;
      grads[idx] = final_gradient;
    }
    break;

    default:
      Logger::panic("Unknown optimizer type:", (int)optimiser_type);
    }
  }

  void apply_weight_gradient(double gradient, double learning_rate, bool is_bias, unsigned weight_index, double clipping_scale, OptimiserType optimiser_type, unsigned neuron_number)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (is_bias)
    {
      apply_update_to_weight(_b_values, _b_grads, _b_velocities, _b_m1, _b_m2, _b_timesteps, _b_decays, weight_index, gradient, learning_rate, clipping_scale, optimiser_type, neuron_number);
    }
    else
    {
      apply_update_to_weight(_w_values, _w_grads, _w_velocities, _w_m1, _w_m2, _w_timesteps, _w_decays, weight_index, gradient, learning_rate, clipping_scale, optimiser_type, neuron_number);
    }
  }

  [[nodiscard]] const std::vector<Neuron>& get_neurons() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _neurons;
  }

  [[nodiscard]] inline const Neuron& get_neuron(unsigned int neuron_index) const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
#if VALIDATE_DATA == 1
    if (neuron_index >= _neurons.size())
    {
      Logger::panic("Index out of bounds in Layer::get_neuron.");
    }
#endif
    return _neurons[neuron_index];
  }

  [[nodiscard]] inline double get_weight_value(unsigned input_idx, unsigned output_idx) const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
#if VALIDATE_DATA == 1
    if ((input_idx * get_number_output_neurons() + output_idx) >= _w_values.size())
    {
      Logger::panic("Index out of bounds in Layer::get_weight_value.");
    }
#endif
    return _w_values[input_idx * get_number_output_neurons() + output_idx];
  }

  [[nodiscard]] inline const double* get_weights_raw(unsigned input_idx) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
#if VALIDATE_DATA == 1
    if ((input_idx * get_number_output_neurons()) >= _w_values.size())
    {
      Logger::panic("Index out of bounds in Layer::get_weights_raw.");
    }
#endif
    return &_w_values[input_idx * get_number_output_neurons()];
  }

  [[nodiscard]] inline double get_bias_value(unsigned output_idx) const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
#if VALIDATE_DATA == 1
    if (output_idx >= _b_values.size())
    {
      Logger::panic("Index out of bounds in Layer::get_bias_value.");
    }
#endif
    return _b_values[output_idx];
  }

  [[nodiscard]] const std::vector<std::vector<WeightParam>>& get_weight_params() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    static thread_local std::vector<std::vector<WeightParam>> thread_local_weights;
    thread_local_weights.assign(get_number_input_neurons(), std::vector<WeightParam>(get_number_output_neurons(), WeightParam(0, 0, 0, 0)));
    for (unsigned i = 0; i < get_number_input_neurons(); ++i) 
    {
      for (unsigned j = 0; j < get_number_output_neurons(); ++j) 
      {
        const auto idx = i * get_number_output_neurons() + j;
        thread_local_weights[i][j] = WeightParam(
            _w_values[idx], _w_grads[idx], _w_velocities[idx],
            _w_m1[idx], _w_m2[idx], _w_timesteps[idx], _w_decays[idx]
        );
      }
    }
    return thread_local_weights;
  }

  [[nodiscard]] const std::vector<WeightParam>& get_bias_weight_params() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    static thread_local std::vector<WeightParam> thread_local_bias_weights;
    thread_local_bias_weights.resize(get_number_output_neurons(), WeightParam(0, 0, 0, 0));
    for (unsigned j = 0; j < get_number_output_neurons(); ++j)
    {
      thread_local_bias_weights[j] = WeightParam(
          _b_values[j], _b_grads[j], _b_velocities[j],
          _b_m1[j], _b_m2[j], _b_timesteps[j], _b_decays[j]
      );
    }
    return thread_local_bias_weights;
  }

  [[nodiscard]] virtual const std::vector<std::vector<WeightParam>>& get_residual_weight_params() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (_residual_projector != nullptr)
    {
      return _residual_projector->get_weight_params();
    }
    static const std::vector<std::vector<WeightParam>> empty_vec_2d;
    return empty_vec_2d;
  }

  void reset_optimizer_state()
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    std::fill(_w_velocities.begin(), _w_velocities.end(), 0.0);
    std::fill(_w_m1.begin(), _w_m1.end(), 0.0);
    std::fill(_w_m2.begin(), _w_m2.end(), 0.0);
    std::fill(_b_velocities.begin(), _b_velocities.end(), 0.0);
    std::fill(_b_m1.begin(), _b_m1.end(), 0.0);
    std::fill(_b_m2.begin(), _b_m2.end(), 0.0);
  }

  [[nodiscard]] inline virtual bool has_bias() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return !_b_values.empty();
  }

  [[nodiscard]] inline double get_momentum() const noexcept 
  { 
    MYODDWEB_PROFILE_FUNCTION("Layer"); 
#if VALIDATE_DATA == 1
    if (_layer_role == LayerRole::Output)
    {
      Logger::panic("The output layer MUST pass the neuron number to get momentum!");
    }
#endif
    return _momentum; 
  }

  [[nodiscard]] inline virtual double get_momentum(unsigned neuron_number) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _momentum;
  }

  virtual Layer* clone() const = 0;

  [[nodiscard]] inline const std::vector<double>& get_w_values() const noexcept { MYODDWEB_PROFILE_FUNCTION("Layer"); return _w_values; }
  [[nodiscard]] inline const std::vector<double>& get_w_grads() const noexcept { MYODDWEB_PROFILE_FUNCTION("Layer"); return _w_grads; }
  [[nodiscard]] inline const std::vector<double>& get_w_velocities() const noexcept { MYODDWEB_PROFILE_FUNCTION("Layer"); return _w_velocities; }
  [[nodiscard]] inline const std::vector<double>& get_w_m1() const noexcept { MYODDWEB_PROFILE_FUNCTION("Layer"); return _w_m1; }
  [[nodiscard]] inline const std::vector<double>& get_w_m2() const noexcept { MYODDWEB_PROFILE_FUNCTION("Layer"); return _w_m2; }
  [[nodiscard]] inline const std::vector<long long>& get_w_timesteps() const noexcept { MYODDWEB_PROFILE_FUNCTION("Layer"); return _w_timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_w_decays() const noexcept { MYODDWEB_PROFILE_FUNCTION("Layer"); return _w_decays; }
  [[nodiscard]] inline const std::vector<double>& get_b_values() const noexcept { MYODDWEB_PROFILE_FUNCTION("Layer"); return _b_values; }
  [[nodiscard]] inline const std::vector<double>& get_b_grads() const noexcept { MYODDWEB_PROFILE_FUNCTION("Layer"); return _b_grads; }
  [[nodiscard]] inline const std::vector<double>& get_b_velocities() const noexcept { MYODDWEB_PROFILE_FUNCTION("Layer"); return _b_velocities; }
  [[nodiscard]] inline const std::vector<double>& get_b_m1() const noexcept { MYODDWEB_PROFILE_FUNCTION("Layer"); return _b_m1; }
  [[nodiscard]] inline const std::vector<double>& get_b_m2() const noexcept { MYODDWEB_PROFILE_FUNCTION("Layer"); return _b_m2; }
  [[nodiscard]] inline const std::vector<long long>& get_b_timesteps() const noexcept { MYODDWEB_PROFILE_FUNCTION("Layer"); return _b_timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_b_decays() const noexcept { MYODDWEB_PROFILE_FUNCTION("Layer"); return _b_decays; }
  
  void set_w_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _w_values = v;
  }
  void set_w_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _w_grads = v;
  }
  void set_w_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _w_velocities = v;
  }
  void set_w_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _w_m1 = v;
  }
  void set_w_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _w_m2 = v;
  }
  void set_w_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _w_timesteps = v;
  }
  void set_w_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _w_decays = v;
  }

  void set_b_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _b_values = v;
  }
  void set_b_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _b_grads = v;
  }
  void set_b_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _b_velocities = v;
  }
  void set_b_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _b_m1 = v;
  }
  void set_b_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _b_m2 = v;
  }
  void set_b_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _b_timesteps = v;
  }
  void set_b_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    _b_decays = v;
  }

  virtual void set_rw_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    (void)v;
  }
  virtual void set_rw_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    (void)v;
  }
  virtual void set_rw_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    (void)v;
  }
  virtual void set_rw_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    (void)v;
  }
  virtual void set_rw_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    (void)v;
  }
  virtual void set_rw_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    (void)v;
  }
  virtual void set_rw_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    (void)v;
  }

  void set_residual_projector(ResidualProjector* p)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    delete _residual_projector;
    _residual_projector = p;
  }

protected:
  Layer(
    unsigned layer_index,
    Role layer_role,
    const layer_activation_helper& lah,
    OptimiserType optimiser_type,
    int residual_layer_number,
    const std::vector<Neuron>& neurons,
    bool has_bias,
    const std::vector<double>& weight_decays,
    ResidualProjector* residual_projector,
    int number_of_threads,
    double momentum
  ) :
    _layer_index(layer_index),
    _layer_role(layer_role),
    _optimiser_type(optimiser_type),
    _residual_layer_number(residual_layer_number),
    _neurons(neurons),
    _w_decays(weight_decays),
    _residual_projector(residual_projector),
    _task_queue_pool(std::make_unique<TaskQueuePool<void>>(number_of_threads)),
    _layer_activation_helper(lah),
    _momentum(momentum)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    const auto& number_input_neurons = lah.get_number_input_neurons();
    const auto& number_output_neurons = lah.get_number_output_neurons();
    const size_t num_weights = static_cast<size_t>(number_input_neurons) * number_output_neurons;
    if (_w_decays.size() != num_weights)
    {
      Logger::panic("The number of weight decays does not match the number of weights.");
    }

    if (number_input_neurons > 0)
    {
      _w_values.resize(num_weights);
      for (size_t i = 0; i < number_input_neurons; ++i) 
      {
        for (size_t j = 0; j < number_output_neurons; ++j) 
        {
          _w_values[i * number_output_neurons + j] = lah.weight_initialization(static_cast<unsigned>(j));
        }
      }
      _w_grads.assign(num_weights, 0.0);
      _w_velocities.assign(num_weights, 0.0);
      _w_m1.assign(num_weights, 0.0);
      _w_m2.assign(num_weights, 0.0);
      _w_timesteps.assign(num_weights, 0);
    }
    if (has_bias)
    {
      _b_values.assign(number_output_neurons, 0.0);
      _b_grads.assign(number_output_neurons, 0.0);
      _b_velocities.assign(number_output_neurons, 0.0);
      _b_m1.assign(number_output_neurons, 0.0);
      _b_m2.assign(number_output_neurons, 0.0);
      _b_timesteps.assign(number_output_neurons, 0);
      _b_decays.assign(number_output_neurons, 0.0);
    }
  }

  Layer(
    unsigned layer_index,
    Role layer_role,
    const activation& activation_method,
    OptimiserType optimiser_type,
    int residual_layer_number,
    unsigned number_input_neurons,
    unsigned number_output_neurons,
    const std::vector<Neuron>& neurons,
    bool has_bias,
    const std::vector<double>& weight_decays,
    ResidualProjector* residual_projector,
    int number_of_threads,
    double momentum
  ) :
    Layer
    (
      layer_index,
      layer_role,
      layer_activation_helper(activation_method, number_input_neurons, number_output_neurons),
      optimiser_type,
      residual_layer_number,
      neurons,
      has_bias,
      weight_decays,
      residual_projector,
      number_of_threads,
      momentum
    )
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
  }

  Layer(
    unsigned layer_index,
    Role layer_role,
    const activation& activation_method,
    OptimiserType optimiser_type,
    int residual_layer_number,
    unsigned number_input_neurons,
    unsigned number_output_neurons,
    const std::vector<Neuron>& neurons,
    bool has_bias,
    double weight_decay,
    ResidualProjector* residual_projector,
    int number_of_threads,
    double momentum
  ) : Layer(
    layer_index,
    layer_role,
    activation_method,
    optimiser_type,
    residual_layer_number,
    number_input_neurons,
    number_output_neurons,
    neurons,
    has_bias,
    std::vector<double>(static_cast<size_t>(number_input_neurons) * number_output_neurons, weight_decay),
    residual_projector,
    number_of_threads,
    momentum
  )
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
  }

  Layer(
    unsigned layer_index,
    const Role layer_role,
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
    const ResidualProjector* residual_projector,
    int number_of_threads,
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
      layer_activation_helper(activation_method, number_input_neurons, number_output_neurons),
      momentum
    )
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
  }

  Layer(
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
    const ResidualProjector* residual_projector,
    int number_of_threads,
    const layer_activation_helper& lah,
    double momentum
  ) noexcept :
    _layer_index(layer_index),
    _layer_role(layer_role),
    _optimiser_type(optimiser_type),
    _residual_layer_number(residual_layer_number),
    _neurons(neurons),
    _w_values(w_values),
    _w_grads(w_grads),
    _w_velocities(w_velocities),
    _w_m1(w_m1),
    _w_m2(w_m2),
    _w_timesteps(w_timesteps),
    _w_decays(w_decays),
    _b_values(b_values),
    _b_grads(b_grads),
    _b_velocities(b_velocities),
    _b_m1(b_m1),
    _b_m2(b_m2),
    _b_timesteps(b_timesteps),
    _b_decays(b_decays),
    _residual_projector(nullptr),
    _task_queue_pool(std::make_unique<TaskQueuePool<void>>(number_of_threads)),
    _layer_activation_helper(lah),
    _momentum(momentum)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (residual_projector != nullptr)
    {
      _residual_projector = new ResidualProjector(*residual_projector);
    }
  }

  static std::vector<Neuron> create_neurons(double dropout_rate, unsigned number_output_neurons)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    std::vector<Neuron> neurons;
    neurons.reserve(number_output_neurons);
    for (unsigned n = 0; n < number_output_neurons; ++n)
    {
      neurons.emplace_back(n, dropout_rate <= 0.0 ? Neuron::Type::Normal : Neuron::Type::Dropout, dropout_rate);
    }
    return neurons;
  }

  unsigned _layer_index;
  Role _layer_role;
  OptimiserType _optimiser_type;
  int _residual_layer_number;
  std::vector<Neuron> _neurons;
  std::vector<double> _w_values, _w_grads, _w_velocities, _w_m1, _w_m2, _w_decays;
  std::vector<long long> _w_timesteps;
  std::vector<double> _b_values, _b_grads, _b_velocities, _b_m1, _b_m2, _b_decays;
  std::vector<long long> _b_timesteps;
  ResidualProjector* _residual_projector;
  std::unique_ptr<TaskQueuePool<void>> _task_queue_pool;
  layer_activation_helper _layer_activation_helper;
  double _momentum;

private:
  [[nodiscard]] inline bool is_bias_index(const std::vector<double>& values) const noexcept
  { 
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return &values == &_b_values; 
  }

  void calculate_huber_loss_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    const EvaluationConfig& evaluation_config,
    const activation::method activation_method,
    std::span<Neuron> neurons) const;

  void calculate_huber_direction_loss_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    const EvaluationConfig& evaluation_config,
    const activation::method activation_method,
    std::span<Neuron> neurons) const;

  void calculate_mse_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    const activation::method activation_method,
    std::span<Neuron> neurons) const;

  void calculate_rmse_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    const activation::method activation_method,
    std::span<Neuron> neurons) const;

  void calculate_bce_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    const EvaluationConfig& evaluation_config,
    const activation::method activation_method,
    std::span<Neuron> neurons) const;

  void calculate_cross_entropy_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    const EvaluationConfig& evaluation_config,
    const activation::method activation_method,
    std::span<Neuron> neurons) const;

  void calculate_log_cosh_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    const activation::method activation_method,
    std::span<Neuron> neurons) const;
};