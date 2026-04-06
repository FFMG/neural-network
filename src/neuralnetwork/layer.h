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

class Layer
{
public:
  enum class LayerType
  {
    Input,
    Hidden,
    Output
  };

  Layer(const Layer& src) noexcept :
    _layer_index(src._layer_index),
    _layer_type(src._layer_type),
    _activation(src._activation),
    _optimiser_type(src._optimiser_type),
    _residual_layer_number(src._residual_layer_number),
    _number_input_neurons(src._number_input_neurons),
    _number_output_neurons(src._number_output_neurons),
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
    _inv_num_neurons(src._inv_num_neurons),
    _weights_cache_dirty(true),
    _bias_weights_cache_dirty(true),
    _task_queue_pool(nullptr)
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
    _layer_type(src._layer_type),
    _activation(std::move(src._activation)),
    _optimiser_type(std::move(src._optimiser_type)),
    _number_input_neurons(src._number_input_neurons),
    _number_output_neurons(src._number_output_neurons),
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
    _inv_num_neurons(src._inv_num_neurons),
    _weights_cache_dirty(true),
    _bias_weights_cache_dirty(true),
    _task_queue_pool(std::move(src._task_queue_pool))
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    src._layer_type = LayerType::Input;
    src._layer_index = 0;
    src._optimiser_type = OptimiserType::None;
    src._number_input_neurons = 0;
    src._number_output_neurons = 0;
    src._residual_layer_number = 0;
    src._residual_projector = nullptr;
  }

  Layer& operator=(const Layer& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (this != &src)
    {
      _layer_index = src._layer_index;
      _layer_type = src._layer_type;
      _activation = src._activation;
      _optimiser_type = src._optimiser_type;
      _number_input_neurons = src._number_input_neurons;
      _number_output_neurons = src._number_output_neurons;
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

      _inv_num_neurons = src._inv_num_neurons;

      _residual_layer_number = src._residual_layer_number;
      delete _residual_projector;
      _residual_projector = nullptr;
      if (src._residual_projector != nullptr)
      {
        _residual_projector = new ResidualProjector(*src._residual_projector);
      }
      _weights_cache_dirty = true;
      _bias_weights_cache_dirty = true;

      if (src._task_queue_pool)
      {
        _task_queue_pool = std::make_unique<TaskQueuePool<void>>(src._task_queue_pool->get_number_of_threads());
      }
      else
      {
        _task_queue_pool.reset();
      }
    }
    return *this;
  }

  Layer& operator=(Layer&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (this != &src)
    {
      _layer_index = src._layer_index;
      _layer_type = src._layer_type;
      _activation = std::move(src._activation);
      _optimiser_type = std::move(src._optimiser_type);
      _number_input_neurons = src._number_input_neurons;
      _number_output_neurons = src._number_output_neurons;
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

      _inv_num_neurons = src._inv_num_neurons;

      delete _residual_projector;
      _residual_projector = src._residual_projector;
      _weights_cache_dirty = true;
      _bias_weights_cache_dirty = true;
      
      _task_queue_pool = std::move(src._task_queue_pool);

      src._layer_index = 0;
      src._optimiser_type = OptimiserType::None;
      src._number_input_neurons = 0;
      src._number_output_neurons = 0;
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
    const std::vector<std::vector<double>>& predictions,
    const std::vector<std::vector<double>>& checking_outputs
    ) const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    Logger::panic("Only output layers can calculate output metrics!");
  }

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

  [[nodiscard]] inline LayerType get_layer_type() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_type;
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

  [[nodiscard]] inline unsigned get_number_input_neurons() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _number_input_neurons;
  }

  [[nodiscard]] inline unsigned get_number_output_neurons() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _number_output_neurons;
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
    unsigned start_neuron,
    unsigned end_neuron) const;

  [[nodiscard]] virtual inline const activation& get_activation() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _activation;
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

  virtual void apply_stored_gradients(double learning_rate, double clipping_scale) = 0;

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
    bool is_bias)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    const size_t n = values.size();
    if (n == 0) return;

    switch (_optimiser_type)
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
      double momentum = _activation.momentum();
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
      const double beta1 = 0.9;
      const double beta2 = 0.999;
      const double epsilon = 1e-8;

      bool all_equal = true;
      for (size_t i = 1; i < n; ++i)
      {
        if (timesteps[i] != timesteps[0])
        {
          all_equal = false;
          break;
        }
      }

      if (all_equal)
      {
        timesteps[0]++;
        const long long t = timesteps[0];
        for (size_t i = 1; i < n; ++i) timesteps[i] = t;

        const double p1 = 1.0 - std::pow(beta1, t);
        const double p2 = 1.0 - std::pow(beta2, t);

        for (size_t i = 0; i < n; ++i)
        {
          double grad = grads[i] * clipping_scale;
          m1[i] = beta1 * m1[i] + (1.0 - beta1) * grad;
          m2[i] = beta2 * m2[i] + (1.0 - beta2) * (grad * grad);

          double m_hat = m1[i] / p1;
          double v_hat = m2[i] / p2;
          double update_step = m_hat / (std::sqrt(v_hat) + epsilon);

          double current_weight = values[i];
          if (_optimiser_type == OptimiserType::AdamW && !is_bias && i < decays.size())
          {
            current_weight *= (1.0 - learning_rate * decays[i]);
          }
          values[i] = current_weight - learning_rate * update_step;
          grads[i] = grad;
        }
      }
      else
      {
        for (size_t i = 0; i < n; ++i)
        {
          double grad = grads[i] * clipping_scale;
          timesteps[i]++;
          const long long t = timesteps[i];
          m1[i] = beta1 * m1[i] + (1.0 - beta1) * grad;
          m2[i] = beta2 * m2[i] + (1.0 - beta2) * (grad * grad);

          double m_hat = m1[i] / (1.0 - std::pow(beta1, t));
          double v_hat = m2[i] / (1.0 - std::pow(beta2, t));
          double update_step = m_hat / (std::sqrt(v_hat) + epsilon);

          double current_weight = values[i];
          if (_optimiser_type == OptimiserType::AdamW && !is_bias && i < decays.size())
          {
            current_weight *= (1.0 - learning_rate * decays[i]);
          }
          values[i] = current_weight - learning_rate * update_step;
          grads[i] = grad;
        }
      }
    }
    break;

    case OptimiserType::Nadam:
    case OptimiserType::NadamW:
    {
      const double beta1 = 0.9;
      const double beta2 = 0.999;
      const double epsilon = 1e-8;

      bool all_equal = true;
      for (size_t i = 1; i < n; ++i)
      {
        if (timesteps[i] != timesteps[0])
        {
          all_equal = false;
          break;
        }
      }

      if (all_equal)
      {
        timesteps[0]++;
        const long long t = timesteps[0];
        for (size_t i = 1; i < n; ++i) timesteps[i] = t;

        const double p1 = 1.0 - std::pow(beta1, t);
        const double p2 = 1.0 - std::pow(beta2, t);

        for (size_t i = 0; i < n; ++i)
        {
          double grad = grads[i] * clipping_scale;
          m1[i] = beta1 * m1[i] + (1.0 - beta1) * grad;
          m2[i] = beta2 * m2[i] + (1.0 - beta2) * (grad * grad);

          double m_hat = m1[i] / p1;
          double v_hat = m2[i] / p2;

          double m_nadam = beta1 * m_hat + ((1.0 - beta1) * grad) / p1;
          double update_step = m_nadam / (std::sqrt(v_hat) + epsilon);

          double current_weight = values[i];
          if (_optimiser_type == OptimiserType::NadamW && !is_bias && i < decays.size())
          {
            current_weight *= (1.0 - learning_rate * decays[i]);
          }
          values[i] = current_weight - learning_rate * update_step;
          grads[i] = grad;
        }
      }
      else
      {
        for (size_t i = 0; i < n; ++i)
        {
          double grad = grads[i] * clipping_scale;
          timesteps[i]++;
          const long long t = timesteps[i];
          m1[i] = beta1 * m1[i] + (1.0 - beta1) * grad;
          m2[i] = beta2 * m2[i] + (1.0 - beta2) * (grad * grad);

          double m_hat = m1[i] / (1.0 - std::pow(beta1, t));
          double v_hat = m2[i] / (1.0 - std::pow(beta2, t));

          double m_nadam = beta1 * m_hat + ((1.0 - beta1) * grad) / (1.0 - std::pow(beta1, t));
          double update_step = m_nadam / (std::sqrt(v_hat) + epsilon);

          double current_weight = values[i];
          if (_optimiser_type == OptimiserType::NadamW && !is_bias && i < decays.size())
          {
            current_weight *= (1.0 - learning_rate * decays[i]);
          }
          values[i] = current_weight - learning_rate * update_step;
          grads[i] = grad;
        }
      }
    }
    break;

    default:
      Logger::panic("Unknown optimizer type:", (int)_optimiser_type);
    }

    _weights_cache_dirty = true;
    _bias_weights_cache_dirty = true;
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
    double clipping_scale)
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

    // Log trace for some updates to avoid flooding
    if (idx == 0 && (timesteps.empty() || timesteps[idx] % 100 == 0))
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

    switch (_optimiser_type)
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
      double velocity = _activation.momentum() * previous_velocity + grad;
      values[idx] -= learning_rate * velocity;
      velocities[idx] = velocity;
      grads[idx] = final_gradient;
    }
    break;

    case OptimiserType::Adam:
    case OptimiserType::AdamW:
    {
      const double beta1 = 0.9;
      const double beta2 = 0.999;
      const double epsilon = 1e-8;

      timesteps[idx]++;
      const auto& time_step = timesteps[idx];

      m1[idx] = beta1 * m1[idx] + (1.0 - beta1) * final_gradient;
      m2[idx] = beta2 * m2[idx] + (1.0 - beta2) * (final_gradient * final_gradient);

      double m_hat = m1[idx] / (1.0 - std::pow(beta1, time_step));
      double v_hat = m2[idx] / (1.0 - std::pow(beta2, time_step));

      double update_step = m_hat / (std::sqrt(v_hat) + epsilon);

      double current_weight = values[idx];
      if (_optimiser_type == OptimiserType::AdamW && !is_bias_index(values) && decays.size() > idx)
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
      const double beta1 = 0.9;
      const double beta2 = 0.999;
      const double epsilon = 1e-8;

      timesteps[idx]++;
      const auto& time_step = timesteps[idx];

      m1[idx] = beta1 * m1[idx] + (1.0 - beta1) * final_gradient;
      m2[idx] = beta2 * m2[idx] + (1.0 - beta2) * (final_gradient * final_gradient);

      double m_hat = m1[idx] / (1.0 - std::pow(beta1, time_step));
      double v_hat = m2[idx] / (1.0 - std::pow(beta2, time_step));

      double m_nadam = beta1 * m_hat + ((1.0 - beta1) * final_gradient) / (1.0 - std::pow(beta1, time_step));
      double update_step = m_nadam / (std::sqrt(v_hat) + epsilon);

      double current_weight = values[idx];
      if (_optimiser_type == OptimiserType::NadamW && !is_bias_index(values) && decays.size() > idx)
      {
        current_weight *= (1.0 - learning_rate * decays[idx]);
      }

      values[idx] = current_weight - learning_rate * update_step;
      grads[idx] = final_gradient;
    }
    break;

    default:
      Logger::panic("Unknown optimizer type:", (int)_optimiser_type);
    }

    _weights_cache_dirty = true;
    _bias_weights_cache_dirty = true;
  }

  void apply_weight_gradient(double gradient, double learning_rate, bool is_bias, unsigned weight_index, double clipping_scale)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (is_bias)
    {
      apply_update_to_weight(_b_values, _b_grads, _b_velocities, _b_m1, _b_m2, _b_timesteps, _b_decays, weight_index, gradient, learning_rate, clipping_scale);
    }
    else
    {
      apply_update_to_weight(_w_values, _w_grads, _w_velocities, _w_m1, _w_m2, _w_timesteps, _w_decays, weight_index, gradient, learning_rate, clipping_scale);
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
    if ((input_idx * _number_output_neurons + output_idx) >= _w_values.size())
    {
      Logger::panic("Index out of bounds in Layer::get_weight_value.");
    }
#endif
    return _w_values[input_idx * _number_output_neurons + output_idx];
  }

  [[nodiscard]] inline const double* get_weights_raw(unsigned input_idx) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
#if VALIDATE_DATA == 1
    if ((input_idx * _number_output_neurons) >= _w_values.size())
    {
      Logger::panic("Index out of bounds in Layer::get_weights_raw.");
    }
#endif
    return &_w_values[input_idx * _number_output_neurons];
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
    if (_weights_cache_dirty) 
    {
      _cached_weights.assign(_number_input_neurons, std::vector<WeightParam>(_number_output_neurons, WeightParam(0,0,0,0)));
      for (unsigned i = 0; i < _number_input_neurons; ++i) {
        for (unsigned j = 0; j < _number_output_neurons; ++j) {
          const auto idx = i * _number_output_neurons + j;
          _cached_weights[i][j] = WeightParam(
            _w_values[idx], _w_grads[idx], _w_velocities[idx],
            _w_m1[idx], _w_m2[idx], _w_timesteps[idx], _w_decays[idx]
          );
        }
      }
      _weights_cache_dirty = false;
    }
    return _cached_weights;
  }

  [[nodiscard]] const std::vector<WeightParam>& get_bias_weight_params() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (_bias_weights_cache_dirty) 
    {
      _cached_bias_weights.resize(_number_output_neurons, WeightParam(0,0,0,0));
      for (unsigned j = 0; j < _number_output_neurons; ++j) 
      {
        _cached_bias_weights[j] = WeightParam(
          _b_values[j], _b_grads[j], _b_velocities[j],
          _b_m1[j], _b_m2[j], _b_timesteps[j], _b_decays[j]
        );
      }
      _bias_weights_cache_dirty = false;
    }
    return _cached_bias_weights;
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

  [[nodiscard]] virtual std::vector<std::vector<WeightParam>>& get_residual_weight_params()
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (_residual_projector != nullptr)
    {
      return const_cast<std::vector<std::vector<WeightParam>>&>(_residual_projector->get_weight_params());
    }
    static std::vector<std::vector<WeightParam>> empty_vec_2d;
    return empty_vec_2d;
  }

  virtual bool has_bias() const noexcept = 0;
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

protected:
  Layer(
    unsigned layer_index,
    LayerType layer_type,
    const activation& activation_method,
    OptimiserType optimiser_type,
    int residual_layer_number,
    unsigned number_input_neurons,
    unsigned number_output_neurons,
    const std::vector<Neuron>& neurons,
    bool has_bias,
    double weight_decay,
    ResidualProjector* residual_projector,
    int number_of_threads
  ) noexcept :
    _layer_index(layer_index),
    _layer_type(layer_type),
    _activation(activation_method),
    _optimiser_type(optimiser_type),
    _residual_layer_number(residual_layer_number),
    _number_input_neurons(number_input_neurons),
    _number_output_neurons(number_output_neurons),
    _neurons(neurons),
    _residual_projector(residual_projector),
    _inv_num_neurons(number_output_neurons > 0 ? 1.0 / number_output_neurons : 0.0),
    _weights_cache_dirty(true),
    _bias_weights_cache_dirty(true),
    _task_queue_pool(std::make_unique<TaskQueuePool<void>>(number_of_threads))
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (number_input_neurons > 0) 
    {
      const size_t num_weights = static_cast<size_t>(number_input_neurons) * number_output_neurons;
      _w_values.resize(num_weights);
      auto initial_weights = _activation.weight_initialization(number_output_neurons, number_input_neurons);
      for (size_t i = 0; i < number_input_neurons; ++i) {
        for (size_t j = 0; j < number_output_neurons; ++j) {
          _w_values[i * number_output_neurons + j] = initial_weights[j];
        }
      }
      _w_grads.assign(num_weights, 0.0);
      _w_velocities.assign(num_weights, 0.0);
      _w_m1.assign(num_weights, 0.0);
      _w_m2.assign(num_weights, 0.0);
      _w_timesteps.assign(num_weights, 0);
      _w_decays.assign(num_weights, weight_decay);
    }
    if (has_bias) 
    {
      _b_values = _activation.weight_initialization(number_output_neurons, 1);
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
    const ResidualProjector* residual_projector,
    int number_of_threads
  ) noexcept :
    _layer_index(layer_index),
    _layer_type(layer_type),
    _activation(activation_method),
    _optimiser_type(optimiser_type),
    _residual_layer_number(residual_layer_number),
    _number_input_neurons(number_input_neurons),
    _number_output_neurons(number_output_neurons),
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
    _inv_num_neurons(number_output_neurons > 0 ? 1.0 / number_output_neurons : 0.0),
    _weights_cache_dirty(true),
    _bias_weights_cache_dirty(true),
    _task_queue_pool(std::make_unique<TaskQueuePool<void>>(number_of_threads))
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (residual_projector != nullptr) { _residual_projector = new ResidualProjector(*residual_projector); }
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
  LayerType _layer_type;
  activation _activation;
  OptimiserType _optimiser_type;
  int _residual_layer_number;
  unsigned _number_input_neurons;
  unsigned _number_output_neurons;
  std::vector<Neuron> _neurons;
  std::vector<double> _w_values, _w_grads, _w_velocities, _w_m1, _w_m2, _w_decays;
  std::vector<long long> _w_timesteps;
  std::vector<double> _b_values, _b_grads, _b_velocities, _b_m1, _b_m2, _b_decays;
  std::vector<long long> _b_timesteps;
  ResidualProjector* _residual_projector;
  std::unique_ptr<TaskQueuePool<void>> _task_queue_pool;
  double _inv_num_neurons;

private:
  bool is_bias_index(const std::vector<double>& values) const noexcept 
  { 
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return &values == &_b_values; 
  }

  void calculate_huber_loss_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    const EvaluationConfig& evaluation_config,
    std::span<Neuron> neurons) const;

  void calculate_huber_direction_loss_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    const EvaluationConfig& evaluation_config,
    std::span<Neuron> neurons) const;

  void calculate_mse_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    std::span<Neuron> neurons) const;

  void calculate_rmse_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    std::span<Neuron> neurons) const;

  void calculate_bce_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    const EvaluationConfig& evaluation_config,
    std::span<Neuron> neurons) const;

  void calculate_cross_entropy_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    std::span<Neuron> neurons) const;

  void calculate_log_cosh_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    std::span<Neuron> neurons) const;

  mutable std::vector<std::vector<WeightParam>> _cached_weights;
  mutable bool _weights_cache_dirty;
  mutable std::vector<WeightParam> _cached_bias_weights;
  mutable bool _bias_weights_cache_dirty;
};