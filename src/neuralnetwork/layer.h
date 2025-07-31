#pragma once
#include "logger.h"
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
      const activation& activation_method,
      const Logger& logger
    )
      : _input_size(input_size),
        _output_size(output_size)
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      _weight_params.reserve(output_size);
      for( unsigned out = 0; out < _output_size; ++out)
      {
        std::vector<WeightParam> weights;
        weights.reserve(_input_size);
        auto values = activation_method.weight_initialization(input_size, 1);  // row: 1 x input_size
        for( auto& value : values)
        {
          weights.emplace_back(WeightParam(value, 0.0, 0.0, logger));
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

    ResidualProjector(const ResidualProjector& rp ) :
       _input_size(rp._input_size),
       _output_size(rp._output_size),
       _weight_params(rp._weight_params)
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    }

    ResidualProjector(ResidualProjector&& rp ) noexcept :
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
          auto value = _weight_params[out][in].value();
          projected[out] += value * residual_layer_outputs[in];
        }
      }
      return projected;
    }

    const std::vector<std::vector<WeightParam>>& get_weights() const 
    { 
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      return _weight_params; 
    }
    inline const std::vector<std::vector<WeightParam>>& get_weight_params() const
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      return _weight_params;
    }
    inline WeightParam& get_weight_params(unsigned residual_source_index, unsigned target_neuron_index)
    { 
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      assert(residual_source_index < _weight_params.size());
      assert(target_neuron_index < _weight_params[residual_source_index].size());
      return _weight_params[residual_source_index][target_neuron_index];
    }
    void update_weight(size_t out, size_t in, double delta) 
    {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      assert(out < _output_size && in < _input_size);
      auto value = _weight_params[out][in].value();
      _weight_params[out][in].set_value(value + delta);
    }

    inline unsigned input_size() const {
      MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
      return _input_size;
    };
    inline unsigned output_size() const {
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
  Layer(unsigned num_neurons_in_previous_layer, unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, int residual_layer_number, LayerType layer_type, const activation::method& activation, const OptimiserType& optimiser_type, double dropout_rate, const Logger& logger);
  Layer(LayerType layer_type, const Logger& logger);

public:  
  Layer(const Layer& src) noexcept;
  Layer(Layer&& src) noexcept;
  Layer& operator=(const Layer& src) noexcept;
  Layer& operator=(Layer&& src) noexcept;
  virtual ~Layer();

  unsigned number_neurons() const;
  const std::vector<Neuron>& get_neurons() const;
  std::vector<Neuron>& get_neurons();

  const Neuron& get_neuron(unsigned index) const;
  Neuron& get_neuron(unsigned index);

  LayerType layer_type() const { return _layer_type;}

  static Layer create_input_layer(const std::vector<Neuron>& neurons, const Logger& logger);
  static Layer create_input_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, const Logger& logger);

  static Layer create_hidden_layer(const std::vector<Neuron>& neurons, unsigned num_neurons_in_previous_layer, int residual_layer_number, const std::vector<std::vector<WeightParam>>& residual_weight_params, const Logger& logger);
  static Layer create_hidden_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, const Layer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, int residual_layer_number, double dropout_rate, const Logger& logger);

  static Layer create_output_layer(const std::vector<Neuron>& neurons, unsigned num_neurons_in_previous_layer, int residual_layer_number, const std::vector<std::vector<WeightParam>>& residual_weight_params, const Logger& logger);
  static Layer create_output_layer(unsigned num_neurons_in_this_layer, const Layer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, int residual_layer_number, const Logger& logger);

  inline int residual_layer_number() const
  { 
    MYODDWEB_PROFILE_FUNCTION("Layer");
    return _residual_layer_number;
  };
  
  inline int residual_input_size() const{
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if(_residual_projector == nullptr)
    {
      return -1;
    }
    return _residual_projector->input_size();
  }
  inline int residual_output_size() const{
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if(_residual_projector == nullptr)
    {
      return -1;
    }
    return _residual_projector->output_size();
  }
  std::vector<double> residual_output_values(const std::vector<double>& residual_layer_outputs) const;
  WeightParam& residual_weight_param(unsigned residual_source_index, unsigned target_neuron_index);
  const std::vector<std::vector<WeightParam>>& residual_weight_params() const;
private:
  void clean();

  std::vector<Neuron> _neurons;
  unsigned _number_input_neurons;  //  number of neurons in previous layer
  unsigned _number_output_neurons; //  number of neurons in this layer
  int _residual_layer_number;
  ResidualProjector* _residual_projector;
  LayerType _layer_type;
  Logger _logger;
};
