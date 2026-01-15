#pragma once

#include "./libraries/instrumentor.h"

#include "activation.h"
#include "errorcalculation.h"
#include "gradientsandoutputs.h"
#include "hiddenstates.h"
#include "hiddenstate.h"
#include "neuron.h"
#include "optimiser.h"
#include "residualprojector.h"
#include "taskqueue.h"
#include "weightparam.h"

#include <cmath>
#include <memory>
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
    if (number_output_neurons == 0)
    {
      Logger::panic("Error: Creating a layer with 0 neurons.");
    }
    if (layer_type != LayerType::Input && number_input_neurons == 0)
    {
      // This is not always a problem, e.g. for a bias-only layer.
      Logger::warning("Warning: Non-input layer created with 0 inputs.");
    }

    // Initialize weights
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
    
    // Initialize biases
    if (has_bias) 
    {
      _b_values = _activation.weight_initialization(number_output_neurons, 1);
      _b_grads.assign(number_output_neurons, 0.0);
      _b_velocities.assign(number_output_neurons, 0.0);
      _b_m1.assign(number_output_neurons, 0.0);
      _b_m2.assign(number_output_neurons, 0.0);
      _b_timesteps.assign(number_output_neurons, 0);
      _b_decays.assign(number_output_neurons, 0.0); // No weight decay for biases
    }
  }

  Layer(
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
    const ResidualProjector* residual_projector,
    int number_of_threads
  ) noexcept :
    _layer_index(layer_index),
    _layer_type(layer_type),
    _activation(activation),
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
    if (residual_projector != nullptr)
    {
      _residual_projector = new ResidualProjector(*residual_projector);
    }
  }

public:
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


  // --- Core Layer Properties ---

  inline OptimiserType get_optimiser_type() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _optimiser_type;
  }

  inline unsigned get_layer_index() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_index;
  }

  inline LayerType get_layer_type() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_type;
  }

  inline int get_residual_layer_number() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _residual_layer_number;
  }

  const ResidualProjector* get_residual_projector() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _residual_projector;
  }

  ResidualProjector* get_residual_projector()
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _residual_projector;
  }

  inline unsigned get_number_neurons() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return get_number_output_neurons();
  }

  inline unsigned get_number_input_neurons() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _number_input_neurons;
  }

  inline unsigned get_number_output_neurons() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _number_output_neurons;
  }

  virtual bool use_bptt() const noexcept 
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return false;
  }

  virtual void calculate_forward_feed(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& previous_layer,
    const std::vector<std::vector<double>>& batch_residual_output_values,
    std::vector<HiddenStates>& batch_hidden_states,
    bool is_training) const = 0;

  virtual void calculate_output_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    std::vector<std::vector<double>>::const_iterator target_outputs_begin,
    const std::vector<HiddenStates>& batch_hidden_states,
    ErrorCalculation::type error_calculation_type) const = 0;

  void calculate_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs,
    ErrorCalculation::type error_calculation_type) const;

  void calculate_mse_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs) const;

  void calculate_bce_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs) const;

  void calculate_cross_entropy_error_deltas(
    std::vector<double>& deltas,
    const std::vector<double>& target_outputs,
    const std::vector<double>& given_outputs) const;

  inline const activation& get_activation() const noexcept
  {
    return _activation;
  }

  virtual void calculate_hidden_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& next_layer,
    const std::vector<std::vector<double>>& batch_next_grad_matrix,
    const std::vector<HiddenStates>& batch_hidden_states,
    int bptt_max_ticks) const = 0;

  virtual void calculate_and_store_gradients(
      const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
      const std::vector<HiddenStates>& hidden_states,
      const Layer& previous_layer,
      int bptt_max_ticks) = 0;

  virtual double get_gradient_norm_sq() const = 0;

  virtual void apply_stored_gradients(double learning_rate, double clipping_scale) = 0;

  void apply_weight_gradient(double gradient, double learning_rate, bool is_bias, unsigned weight_index, double clipping_scale)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
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
    switch (_optimiser_type)
    {
    case OptimiserType::None:
      apply_none_update(final_gradient, learning_rate, is_bias, weight_index);
      break;

    case OptimiserType::SGD:
      apply_sgd_update(final_gradient, learning_rate, _activation.momentum(), is_bias, weight_index);
      break;

    case OptimiserType::Adam:
      apply_adam_update(final_gradient, learning_rate, 0.9, 0.999, 1e-8, is_bias, weight_index);
      break;

    case OptimiserType::AdamW:
      apply_adamw_update(final_gradient, learning_rate, 0.9, 0.999, 1e-8, is_bias, weight_index);
      break;

    case OptimiserType::Nadam:
      apply_nadam_update(final_gradient, learning_rate, 0.9, 0.999, 1e-8, is_bias, weight_index);
      break;

    case OptimiserType::NadamW:
      apply_nadamw_update(final_gradient, learning_rate, 0.9, 0.999, 1e-8, is_bias, weight_index);
      break;

    default:
      Logger::panic("Unknown optimizer type:", (int)_optimiser_type);
    }
  }

  const std::vector<Neuron>& get_neurons() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _neurons;
  }

  const Neuron& get_neuron(unsigned int neuron_index) const
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

  // --- Weights and Biases ---

  inline double get_weight_value(unsigned input_idx, unsigned output_idx) const 
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

  inline double get_bias_value(unsigned output_idx) const 
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

  // This is for serializer compatibility. It's slow.
  const std::vector<std::vector<WeightParam>>& get_weight_params() const
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

  const std::vector<WeightParam>& get_bias_weight_params() const
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
  
  virtual const std::vector<std::vector<WeightParam>>& get_residual_weight_params() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (_residual_projector != nullptr)
    {
      return _residual_projector->get_weight_params();
    }
    static const std::vector<std::vector<WeightParam>> empty_vec_2d;
    return empty_vec_2d;
  }

  virtual std::vector<std::vector<WeightParam>>& get_residual_weight_params()
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

  inline const std::vector<double>& get_w_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _w_values;
  }
  inline const std::vector<double>& get_w_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _w_grads;
  }
  inline const std::vector<double>& get_w_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _w_velocities;
  }
  inline const std::vector<double>& get_w_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _w_m1;
  }
  inline const std::vector<double>& get_w_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _w_m2;
  }
  inline const std::vector<long long>& get_w_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _w_timesteps;
  }
  inline const std::vector<double>& get_w_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _w_decays;
  }
  inline const std::vector<double>& get_b_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _b_values;
  }
  inline const std::vector<double>& get_b_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _b_grads;
  }
  inline const std::vector<double>& get_b_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _b_velocities;
  }
  inline const std::vector<double>& get_b_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _b_m1;
  }
  inline const std::vector<double>& get_b_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _b_m2;
  }
  inline const std::vector<long long> get_b_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _b_timesteps;
  }
  inline const std::vector<double>& get_b_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _b_decays;
  }

protected:
  static std::vector<Neuron> create_neurons(
    double dropout_rate,
    unsigned number_output_neurons
  )
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    std::vector<Neuron> neurons;
    neurons.reserve(number_output_neurons);
    for (unsigned neuron_number = 0; neuron_number < number_output_neurons; ++neuron_number)
    {
      auto neuron = Neuron(
        neuron_number,
        dropout_rate <= 0.0 ? Neuron::Type::Normal : Neuron::Type::Dropout,
        dropout_rate);
      neurons.emplace_back(neuron);
    }
    return neurons;
  }

  inline void apply_none_update(double raw_gradient, double learning_rate, bool is_bias, unsigned idx)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    auto& values = is_bias ? _b_values : _w_values;
    auto& grads = is_bias ? _b_grads : _w_grads;

    double new_weight = values[idx] - learning_rate * raw_gradient;
    grads[idx] = raw_gradient;
    values[idx] = new_weight;
    _weights_cache_dirty = true;
    _bias_weights_cache_dirty = true;
  }

  inline void apply_sgd_update(double raw_gradient, double learning_rate, double momentum, bool is_bias, unsigned idx)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    auto& values = is_bias ? _b_values : _w_values;
    auto& grads = is_bias ? _b_grads : _w_grads;
    auto& velocities = is_bias ? _b_velocities : _w_velocities;
    auto& decays = is_bias ? _b_decays : _w_decays;
    
    if (!is_bias && decays[idx] > 0.0)
    {
      raw_gradient += decays[idx] * values[idx];
    }

    double previous_velocity = velocities[idx];
    double velocity = momentum * previous_velocity + raw_gradient;

    double new_weight = values[idx] - learning_rate * velocity;

    grads[idx] = raw_gradient;
    values[idx] = new_weight;
    velocities[idx] = velocity;
    _weights_cache_dirty = true;
    _bias_weights_cache_dirty = true;
  }

  inline void apply_adam_update(double raw_gradient, double learning_rate, double beta1, double beta2, double epsilon, bool is_bias, unsigned idx)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    auto& values = is_bias ? _b_values : _w_values;
    auto& grads = is_bias ? _b_grads : _w_grads;
    auto& m1s = is_bias ? _b_m1 : _w_m1;
    auto& m2s = is_bias ? _b_m2 : _w_m2;
    auto& timesteps = is_bias ? _b_timesteps : _w_timesteps;
    auto& decays = is_bias ? _b_decays : _w_decays;
    
    timesteps[idx]++;
    const auto& time_step = timesteps[idx];

    m1s[idx] = beta1 * m1s[idx] + (1.0 - beta1) * raw_gradient;
    m2s[idx] = beta2 * m2s[idx] + (1.0 - beta2) * raw_gradient * raw_gradient;

    double m_hat = m1s[idx] / (1.0 - std::pow(beta1, time_step));
    double v_hat = m2s[idx] / (1.0 - std::pow(beta2, time_step));

    double adam_update = learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    
    double decayed_weight = is_bias ? values[idx] : values[idx] * (1.0 - learning_rate * decays[idx]);
    double new_weight = decayed_weight - adam_update;

    values[idx] = new_weight;
    grads[idx] = raw_gradient;
    _weights_cache_dirty = true;
    _bias_weights_cache_dirty = true;
  }

  inline void apply_adamw_update(double raw_gradient, double learning_rate, double beta1, double beta2, double epsilon, bool is_bias, unsigned idx)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    auto& values = is_bias ? _b_values : _w_values;
    auto& grads = is_bias ? _b_grads : _w_grads;
    auto& m1s = is_bias ? _b_m1 : _w_m1;
    auto& m2s = is_bias ? _b_m2 : _w_m2;
    auto& timesteps = is_bias ? _b_timesteps : _w_timesteps;
    auto& decays = is_bias ? _b_decays : _w_decays;

    timesteps[idx]++;
    const auto& time_step = timesteps[idx];

    m1s[idx] = beta1 * m1s[idx] + (1.0 - beta1) * raw_gradient;
    m2s[idx] = beta2 * m2s[idx] + (1.0 - beta2) * (raw_gradient * raw_gradient);

    double m_hat = m1s[idx] / (1.0 - std::pow(beta1, time_step));
    double v_hat = m2s[idx] / (1.0 - std::pow(beta2, time_step));

    double weight_update = learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon));
    
    double new_weight = values[idx];
    if (!is_bias) {
      new_weight *= (1.0 - learning_rate * decays[idx]);
    }

    new_weight -= weight_update;

    values[idx] = new_weight;
    grads[idx] = raw_gradient;
    _weights_cache_dirty = true;
    _bias_weights_cache_dirty = true;
  }

  inline void apply_nadam_update(double raw_gradient, double learning_rate, double beta1, double beta2, double epsilon, bool is_bias, unsigned idx)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    auto& values = is_bias ? _b_values : _w_values;
    auto& grads = is_bias ? _b_grads : _w_grads;
    auto& m1s = is_bias ? _b_m1 : _w_m1;
    auto& m2s = is_bias ? _b_m2 : _w_m2;
    auto& timesteps = is_bias ? _b_timesteps : _w_timesteps;

#if VALIDATE_DATA == 1
    if (idx >= m1s.size())
    {
      Logger::panic("m1s size is invalid in apply_nadam_update!");
    }
    if (idx >= m2s.size())
    {
      Logger::panic("m2s size is invalid in apply_nadam_update!");
    }
    if (idx >= values.size())
    {
      Logger::panic("values size is invalid in apply_nadam_update!");
    }
    if (idx >= grads.size())
    {
      Logger::panic("grads size is invalid in apply_nadam_update!");
    }
    if (idx >= timesteps.size())
    {
      Logger::panic("timesteps size is invalid in apply_nadam_update!");
    }
#endif

    ++timesteps[idx];
    const auto& time_step = timesteps[idx];
    
    m1s[idx] = beta1 * m1s[idx] + (1.0 - beta1) * raw_gradient;
    m2s[idx] = beta2 * m2s[idx] + (1.0 - beta2) * (raw_gradient * raw_gradient);

    double m_hat = m1s[idx] / (1.0 - std::pow(beta1, time_step));
    double v_hat = m2s[idx] / (1.0 - std::pow(beta2, time_step));

    double corrected_gradient = (beta1 * m_hat) + ((1.0 - beta1) * raw_gradient) / (1.0 - std::pow(beta1, time_step));
    double weight_update = learning_rate * (corrected_gradient / (std::sqrt(v_hat) + epsilon));
    
    double new_weight = values[idx] - weight_update;

    values[idx] = new_weight;
    grads[idx] = raw_gradient;
    _weights_cache_dirty = true;
    _bias_weights_cache_dirty = true;
  }

  inline void apply_nadamw_update(double raw_gradient, double learning_rate, double beta1, double beta2, double epsilon, bool is_bias, unsigned idx)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    auto& values = is_bias ? _b_values : _w_values;
    auto& grads = is_bias ? _b_grads : _w_grads;
    auto& m1s = is_bias ? _b_m1 : _w_m1;
    auto& m2s = is_bias ? _b_m2 : _w_m2;
    auto& timesteps = is_bias ? _b_timesteps : _w_timesteps;
    auto& decays = is_bias ? _b_decays : _w_decays;

#if VALIDATE_DATA == 1
    if (idx >= m1s.size())
    {
      Logger::panic("m1s size is invalid in apply_nadamw_update!");
    }
    if (idx >= m2s.size())
    {
      Logger::panic("m2s size is invalid in apply_nadamw_update!");
    }
    if (idx >= values.size())
    {
      Logger::panic("values size is invalid in apply_nadamw_update!");
    }
    if (idx >= grads.size())
    {
      Logger::panic("grads size is invalid in apply_nadamw_update!");
    }
    if (idx >= timesteps.size())
    {
      Logger::panic("timesteps size is invalid in apply_nadamw_update!");
    }
#endif
    ++timesteps[idx];
    const long long time_step = timesteps[idx];

    m1s[idx] = beta1 * m1s[idx] + (1.0 - beta1) * raw_gradient;
    m2s[idx] = beta2 * m2s[idx] + (1.0 - beta2) * (raw_gradient * raw_gradient);

    double m_hat = m1s[idx] / (1.0 - std::pow(beta1, time_step));
    double v_hat = m2s[idx] / (1.0 - std::pow(beta2, time_step));

    double m_nadam = beta1 * m_hat + ((1.0 - beta1) * raw_gradient) / (1.0 - std::pow(beta1, time_step));
    double step = m_nadam / (std::sqrt(v_hat) + epsilon);
    
    double w = values[idx];
    if (!is_bias)
    {
      w *= (1.0 - learning_rate * decays[idx]);
    }
    
    w -= learning_rate * step;
    values[idx] = w;
    grads[idx] = raw_gradient;
    _weights_cache_dirty = true;
    _bias_weights_cache_dirty = true;
  }

protected:
  /**
   * All the values below need to be serialised
   */

  unsigned _layer_index;
  LayerType _layer_type;
  activation _activation;
  OptimiserType _optimiser_type;
  int _residual_layer_number;

  unsigned _number_input_neurons;
  unsigned _number_output_neurons;

  std::vector<Neuron> _neurons;
  
  // Structure of Arrays (SoA) for weights
  std::vector<double> _w_values;
  std::vector<double> _w_grads;
  std::vector<double> _w_velocities;
  std::vector<double> _w_m1; // first moment
  std::vector<double> _w_m2; // second moment
  std::vector<long long> _w_timesteps;
  std::vector<double> _w_decays;

  // Structure of Arrays (SoA) for biases
  std::vector<double> _b_values;
  std::vector<double> _b_grads;
  std::vector<double> _b_velocities;
  std::vector<double> _b_m1;
  std::vector<double> _b_m2;
  std::vector<long long> _b_timesteps;
  std::vector<double> _b_decays;
  
  ResidualProjector* _residual_projector;

  std::unique_ptr<TaskQueuePool<void>> _task_queue_pool;

  /**
   * All the values below do not need to be serialised
   */
  double _inv_num_neurons;

private:
  // Caches for serializer compatibility
  mutable std::vector<std::vector<WeightParam>> _cached_weights;
  mutable bool _weights_cache_dirty;
  mutable std::vector<WeightParam> _cached_bias_weights;
  mutable bool _bias_weights_cache_dirty;
};