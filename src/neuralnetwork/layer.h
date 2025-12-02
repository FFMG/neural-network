#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif

#include "activation.h"
#include "errorcalculation.h"
#include "hiddenstate.h"
#include "neuron.h"
#include "optimiser.h"
#include "weightparam.h"

#include <cassert>
#include <vector>

class Neuron;
class Layer
{
protected:
  class ResidualProjector
  {
  public:
    ResidualProjector(
      unsigned input_size,       // size of residual_layer_outputs (e.g., 160)
      unsigned output_size,      // size of the target layer (e.g., 128)
      const activation& activation_method
    )
      : _input_size(input_size),
      _output_size(output_size)
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      _weight_params.reserve(output_size);
      for (unsigned out = 0; out < _output_size; ++out)
      {
        std::vector<WeightParam> weights;
        weights.reserve(_input_size);
        auto values = activation_method.weight_initialization(input_size, 1);  // row: 1 x input_size
        for (auto& value : values)
        {
          // TODO set weigh_decay in the options and add it to the info output on start of training.
          weights.emplace_back(WeightParam(value, 0.0, 0.0, 0.0, 0.0));
        }
        _weight_params.emplace_back(weights);
      }
    }

    ResidualProjector(const std::vector<std::vector<WeightParam>>& weight_params) :
      _input_size(0),
      _output_size(0),
      _weight_params(weight_params)
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      _output_size = static_cast<unsigned>(weight_params.size());
      _input_size = _output_size > 0 ? static_cast<unsigned>(weight_params.back().size()) : 0;
    }

    ResidualProjector(const ResidualProjector& rp) :
      _input_size(rp._input_size),
      _output_size(rp._output_size),
      _weight_params(rp._weight_params)
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    }

    ResidualProjector(ResidualProjector&& rp) noexcept :
      _input_size(rp._input_size),
      _output_size(rp._output_size),
      _weight_params(std::move(rp._weight_params))
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    }

    virtual ~ResidualProjector() = default;

    // Projects residual_layer_outputs (size = input_size) to a vector of size = output_size
    std::vector<double> project(const std::vector<double>& residual_layer_outputs) const
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      assert(residual_layer_outputs.size() == _input_size);
      std::vector<double> projected(_output_size, 0.0);
      for (size_t out = 0; out < _output_size; ++out)
      {
        for (size_t in = 0; in < _input_size; ++in)
        {
          auto value = _weight_params[out][in].get_value();
          projected[out] += value * residual_layer_outputs[in];
        }
      }
      return projected;
    }
    inline const std::vector<std::vector<WeightParam>>& get_weight_params() const noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      return _weight_params;
    }
    inline std::vector<std::vector<WeightParam>>& get_weight_params() noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      return _weight_params;
    }
    inline std::vector<WeightParam>& get_weight_params(unsigned neuron_index) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      return _weight_params[neuron_index];
    }
    inline WeightParam& get_weight_params(unsigned residual_source_index, unsigned target_neuron_index) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      assert(target_neuron_index < _weight_params.size());
      assert(residual_source_index < _weight_params[target_neuron_index].size());
      return _weight_params[target_neuron_index][residual_source_index];
    }
    inline void update_weight(size_t out, size_t in, double delta) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      assert(out < _output_size && in < _input_size);
      auto value = _weight_params[out][in].get_value();
      _weight_params[out][in].set_value(value + delta);
    }

    inline unsigned input_size() const noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      return _input_size;
    };
    inline unsigned output_size() const noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      return _output_size;
    };

  private:
    unsigned _input_size;
    unsigned _output_size;
    std::vector<std::vector<WeightParam>> _weight_params;  // shape: [output][input]
  };

  void move_residual_projector(ResidualProjector* residual_projector);

  friend class Layers;
public:
  enum class LayerType
  {
    Input,
    Hidden,
    Output
  };
private:
  Layer(unsigned layer_index, 
    unsigned num_neurons_in_previous_layer, 
    unsigned num_neurons_in_this_layer, 
    unsigned num_neurons_in_next_layer, 
    double weight_decay,
    int residual_layer_number, 
    LayerType layer_type, 
    const activation::method& activation_method, 
    const OptimiserType& optimiser_type, 
    double dropout_rate);

public:
  Layer(
    unsigned layer_index,
    const std::vector<Neuron>& neurons,
    unsigned number_input_neurons,
    int residual_layer_number,
    LayerType layer_type,
    OptimiserType optimiser_type,
    const activation::method& activation_method,
    const std::vector<std::vector<WeightParam>>& weights,
    const std::vector<WeightParam>& bias_weights,
    const std::vector<std::vector<WeightParam>>& residual_weights
    );

  Layer(const Layer& src) noexcept;
  Layer(Layer&& src) noexcept;
  Layer& operator=(const Layer& src) noexcept;
  Layer& operator=(Layer&& src) noexcept;
  virtual ~Layer();

  unsigned number_neurons() const noexcept;
  unsigned number_neurons_with_bias() const noexcept;
  const std::vector<Neuron>& get_neurons() const noexcept;
  std::vector<Neuron>& get_neurons() noexcept;

  const Neuron& get_neuron(unsigned index) const;
  Neuron& get_neuron(unsigned index);

  LayerType layer_type() const { 
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_type; 
  }

public:
  static Layer create_input_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, double weight_decay );
  static Layer create_hidden_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, double weight_decay, const Layer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, int residual_layer_number, double dropout_rate);
  static Layer create_output_layer(unsigned num_neurons_in_this_layer, double weight_decay, const Layer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, int residual_layer_number);

  inline int residual_layer_number() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _residual_layer_number;
  };

  inline int residual_input_size() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (_residual_projector == nullptr)
    {
      return -1;
    }
    return _residual_projector->input_size();
  }

  inline int residual_output_size() const 
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (_residual_projector == nullptr)
    {
      return -1;
    }
    return _residual_projector->output_size();
  }

  std::vector<std::vector<double>> project_residual_output_values(const std::vector<std::vector<double>>& residual_layer_outputs) const;
  std::vector<double> project_residual_output_values(const std::vector<double>& residual_layer_outputs) const;
  WeightParam& residual_weight_param(unsigned residual_source_index, unsigned target_neuron_index);
  const std::vector<std::vector<WeightParam>>& residual_weight_params() const;

  std::vector<std::vector<double>> calculate_forward_feed(
    const Layer& previous_layer,
    const std::vector<std::vector<double>>& previous_layer_inputs,
    const std::vector<std::vector<double>>& residual_output_values,
    std::vector<std::vector<HiddenState>>& hidden_states,
    bool is_training) const;

  std::vector<std::vector<double>> calculate_output_gradients(
    const std::vector<std::vector<double>>& target_outputs,
    const std::vector<std::vector<double>>& given_outputs,
    const std::vector<std::vector<HiddenState>>& hidden_states,
    double gradient_clip_threshold,
    ErrorCalculation::type error_calculation_type) const;

  void calculate_error_deltas(
    std::vector<std::vector<double>>& deltas,
    const std::vector<std::vector<double>>& target_outputs,
    const std::vector<std::vector<double>>& given_outputs,
    ErrorCalculation::type error_calculation_type) const;

  void calculate_mse_error_deltas(
    std::vector<std::vector<double>>& deltas,
    const std::vector<std::vector<double>>& target_outputs,
    const std::vector<std::vector<double>>& given_outputs) const;

  void calculate_bce_error_deltas(
    std::vector<std::vector<double>>& deltas,
    const std::vector<std::vector<double>>& target_outputs,
    const std::vector<std::vector<double>>& given_outputs) const;

  std::vector<std::vector<double>> calculate_hidden_gradients(
    const Layer& next_layer,
    const std::vector<std::vector<double>>& next_grad_matrix,
    const std::vector<std::vector<double>>& output_matrix,
    const std::vector<std::vector<HiddenState>>& hidden_states,
    double gradient_clip_threshold) const;

  inline unsigned number_input_neurons(bool add_bias) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _number_input_neurons + (add_bias ? 1 : 0);
  }

  inline unsigned get_layer_index() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_index;
  }

  void apply_weight_gradient(const double gradient, const double learning_rate, bool is_bias, WeightParam& weight_param, double clipping_scale, double gradient_clip_threshold);

  // optimisers
  // TODO THOSE SHOULD ALL BE PRIVATE
  static double clip_gradient(double gradient, double gradient_clip_threshold);
  static void apply_none_update(WeightParam& weight_param, double raw_gradient, double learning_rate);
  static void apply_sgd_update(WeightParam& weight_param, double raw_gradient, double learning_rate, double momentum, bool is_bias);
  static void apply_adam_update(WeightParam& weight_param, double raw_gradient, double learning_rate, double beta1, double beta2, double epsilon, bool is_bias);
  static void apply_adamw_update(
    WeightParam& weight_param,
    double raw_gradient,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    bool is_bias
  );
  static void apply_nadam_update(
    WeightParam& weight_param,
    double raw_gradient,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon
  );
  static void apply_nadamw_update(
    WeightParam& weight_param,
    double raw_gradient,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    bool is_bias
  );

  const std::vector<std::vector<WeightParam>>& get_weight_params() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _weights;
  }

  const WeightParam& get_weight_param(unsigned input_neuron_number, unsigned neuron_index) const
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    // TODO Valiate
    return _weights[input_neuron_number][neuron_index];
  }
  WeightParam& get_weight_param(unsigned input_neuron_number, unsigned neuron_index)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    // TODO Valiate
    return _weights[input_neuron_number][neuron_index];
  }
  inline WeightParam& get_bias_weight_param(unsigned neuron_index) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    // TODO Valiate
    return _bias_weights[neuron_index];
  }

  bool has_bias() const noexcept;

  inline const std::vector<WeightParam>& get_bias_weight_params() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    // TODO Valiate and profile.
    return _bias_weights;
  }
  inline const std::vector<std::vector<WeightParam>>& get_residual_weight_params() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (_residual_projector == nullptr)
    {
      static const std::vector<std::vector<WeightParam>> empty;
      return empty;
    }
    return _residual_projector->get_weight_params();
  }
  inline std::vector<std::vector<WeightParam>>& get_residual_weight_params() noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (_residual_projector == nullptr)
    {
      static std::vector<std::vector<WeightParam>> empty;
      return empty;
    }
    return _residual_projector->get_weight_params();
  }
  inline std::vector<WeightParam>& get_residual_weight_params(unsigned neuron_index) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (_residual_projector == nullptr)
    {
      static std::vector<WeightParam> empty;
      return empty;
    }
    // TODO VALIDATE
    return _residual_projector->get_weight_params()[neuron_index];
  }
  inline const OptimiserType get_optimiser_type() const noexcept {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _optimiser_type;
  }
  inline const activation& get_activation() const noexcept {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _activation;
  }
  inline const LayerType& get_layer_type() const noexcept {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _layer_type;
  }
  inline unsigned get_number_input_neurons() const {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _number_input_neurons;
  }
  inline unsigned get_number_output_neurons() const {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _number_output_neurons;
  }

private:
  void clean();
  void resize_weights(
    const activation& activation_method,
    unsigned number_input_neurons, 
    unsigned number_output_neurons, 
    double weight_decay);

  unsigned _layer_index;
  std::vector<Neuron> _neurons;
  unsigned _number_input_neurons;  //  number of neurons in previous layer
  unsigned _number_output_neurons; //  number of neurons in this layer
  int _residual_layer_number;
  ResidualProjector* _residual_projector;
  LayerType _layer_type;
  OptimiserType _optimiser_type;
  activation _activation;

  // N_prev = number of neurons in previous layer
  // N_this = number of neurons in this layer
  // Size: [N_prev][N_this]
  std::vector<std::vector<WeightParam>> _weights;
  std::vector<WeightParam> _bias_weights;
};
