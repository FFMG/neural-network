#pragma once
#include <cassert>
#include <functional>
#include <vector>

#include "activation.h"
#include "layer.h"
#include "neuron.h"

class NeuralNetwork
{
private:
  class LayersAndNeurons
  {
  public:
    LayersAndNeurons() noexcept
    {
    }
    LayersAndNeurons(const LayersAndNeurons& src) noexcept:
      _data(src._data)
    {
    }
    LayersAndNeurons(LayersAndNeurons&& src) noexcept:
      _data(std::move(src._data))
    {
    }
    LayersAndNeurons& operator=(const LayersAndNeurons&src) noexcept
    {
      if(this != &src)
      {
        _data = src._data;
      }
      return *this;
    }
    LayersAndNeurons& operator=(LayersAndNeurons&& src) noexcept
    {
      if(this != &src)
      {
        _data = std::move(src._data);
      }
      return *this;
    }
    LayersAndNeurons(const std::vector<std::vector<double>>& data) noexcept :
      _data(data)
    {
    }
    LayersAndNeurons(std::vector<std::vector<double>>&& data) noexcept :
      _data(std::move(data))
    {
    }
    LayersAndNeurons& operator=(const std::vector<std::vector<double>>& data) noexcept
    {
      _data = data;
      return *this;
    }
    LayersAndNeurons& operator=(std::vector<std::vector<double>>&& data) noexcept
    {
      _data = std::move(data);
      return *this;
    }
    void zero()
    {
      _data = {};
    }

    std::vector<double> get_neurons(size_t layer) const 
    {
      if (layer >= number_layers())
      {
        throw std::out_of_range("LayersAndNeurons::get_row() - layer out of range");
      }
      return _data[layer];
    }

    void set(size_t layer, size_t neuron, double value) 
    {
      ensure_size(layer, neuron);
      _data[layer][neuron] = value;
    }

    double get(size_t layer, size_t neuron) const 
    {
      if (layer >= number_layers())
      {
        throw std::out_of_range("LayersAndNeurons::get() - index out of range");
      }
      if (neuron >= number_neurons(layer))
      {
        if (neuron == number_neurons(layer))
        {
          return 1; //  bias
        }
        throw std::out_of_range("LayersAndNeurons::get(layer) - index out of range");
      }
      return _data[layer][neuron];
    }

    size_t number_layers() const { return _data.size(); }
    size_t number_neurons(size_t layer) const 
    { 
      return layer >= number_layers() ? 0 : _data[layer].size();
    }

  private:
    void ensure_size(size_t layer, size_t neuron)
    {
      if (layer >= _data.size())
      {
        // Resize 'data' to accommodate the new layer index.
        // All new layers will be empty vectors initially.
        _data.resize(layer + 1);
      }      
      if (neuron >= _data[layer].size()) 
      {
        // Resize the specific layer's neuron vector.
        // New neurons are default-initialized to 0.0.
        _data[layer].resize(neuron + 1, 0.0);
      }      
    }
    std::vector<std::vector<double>> _data;
  };

  class AverageGradientsAndOutputs
  {
  public:
    AverageGradientsAndOutputs() noexcept:
      _batch_size(0)
    {

    };
    AverageGradientsAndOutputs(const AverageGradientsAndOutputs& src) noexcept:
      _data(src._data),
      _batch_size(src._batch_size)
    {
    }
    AverageGradientsAndOutputs(AverageGradientsAndOutputs&& src) noexcept
    {
      _data = std::move(src._data);
      _batch_size = src._batch_size;
      src._batch_size = 0;
    }
    AverageGradientsAndOutputs& operator=(const AverageGradientsAndOutputs& src) noexcept
    {
      if( &src != this)
      {
        _data = src._data;
        _batch_size = src._batch_size;
      }
      return *this;
    }
    AverageGradientsAndOutputs& operator=(AverageGradientsAndOutputs&& src) noexcept
    {
      if( &src != this)
      {
        _data = std::move(src._data);
        _batch_size = src._batch_size;
        src._batch_size = 0;
      }
      return *this;
    }

    void add_outputs_gradients(const AverageGradientsAndOutputs& average_outputs_gradients)
    {
      const auto& outputs_gradients = average_outputs_gradients.get_outputs_gradients();
      for(size_t layer = 0; layer < outputs_gradients.size(); ++layer)
      {
        for(size_t neuron = 0; neuron < outputs_gradients[layer].size(); ++neuron)
        {
          ensure_size(layer, neuron);
          size_t number_outputs = outputs_gradients[layer][neuron].size();
          if(_data[layer][neuron].size() == 0 )
          {
            _data[layer][neuron].resize(number_outputs, 0.0);
          }

          for(size_t i = 0; i < number_outputs; ++i)
          {
            _data[layer][neuron][i] += outputs_gradients[layer][neuron][i];
          }
        }
      }
      ++_batch_size;
    }

    void set_outputs_gradients(unsigned layer, unsigned target_neuron, const std::vector<double>& outputs_gradients)
    {
      // if we are calling 'set' then it is for a single batch
      // otherwise call the add_outputs_gradients(...)
      assert(_batch_size== 0 || _batch_size == 1);
      ensure_size(layer, target_neuron);
      _data[layer][target_neuron] = outputs_gradients;
      _batch_size = 1;
    }

    std::vector<double> get_outputs_gradients(unsigned layer, unsigned target_neuron) const
    {
      if(_batch_size == 0 )
      {
        return {};
      }
      if(_data.size() <= layer)
      {
        return {};
      }
      if(_data[layer].size() <= target_neuron)
      {
        return {};
      }
      if(_batch_size == 1)
      {
        return _data[layer][target_neuron];
      }
      const size_t output_size = _data[layer][target_neuron].size();
      std::vector<double> average_output_gradient;
      average_output_gradient.reserve(output_size);
      for( size_t output_number = 0; output_number < output_size; ++output_number )
      {
        const auto& output = _data[layer][target_neuron][output_number];
        average_output_gradient.emplace_back(output / static_cast<double>(_batch_size));
      }
      return average_output_gradient;
    }

  protected:
    void zero_outputs_gradients(){ _data = {};};

    const std::vector<std::vector<std::vector<double>>>& get_outputs_gradients() const
    {
      return _data;
    }

    void ensure_size(size_t layer, size_t neuron)
    {
      if (layer >= _data.size())
      {
        // Resize 'data' to accommodate the new layer index.
        // All new layers will be empty vectors initially.
        _data.resize(layer + 1);
      }      
      if (neuron >= _data[layer].size()) 
      {
        // Resize the specific layer's neuron vector.
        // New neurons are default-initialized to 0.0.
        _data[layer].resize(neuron + 1, {});
      }      
    }

    std::vector<std::vector<std::vector<double>>> _data;
    int _batch_size = 0;
  };

  class GradientsAndOutputs : public AverageGradientsAndOutputs
  {
  public:
    GradientsAndOutputs() : AverageGradientsAndOutputs()
    {
    };
    GradientsAndOutputs(const GradientsAndOutputs& src) : 
      AverageGradientsAndOutputs(src),
      _outputs(src._outputs),
      _gradients(src._gradients)
    {
    };
    GradientsAndOutputs(GradientsAndOutputs&& src) noexcept: 
      AverageGradientsAndOutputs(std::move(src)),
      _outputs(std::move(src._outputs)),
      _gradients(std::move(src._gradients))
    {
    };
    GradientsAndOutputs& operator=(const GradientsAndOutputs& src) noexcept
    {
      if( &src != this)
      {
        AverageGradientsAndOutputs::operator=(src);
        _outputs = src._outputs;
        _gradients = src._gradients;
      }
      return *this;
    }
    GradientsAndOutputs& operator=(GradientsAndOutputs&& src) noexcept
    {
      if( &src != this)
      {
        AverageGradientsAndOutputs::operator=(std::move(src));
        _outputs = std::move(src._outputs);
        _gradients = std::move(src._gradients);
      }
      return *this;
    }
    virtual ~GradientsAndOutputs() = default;

    unsigned num_output_layers() const 
    { 
      return static_cast<unsigned>(_outputs.number_layers());
    }

    unsigned num_output_neurons(unsigned layer) const 
    { 
      return static_cast<unsigned>(_outputs.number_neurons(layer));
    }

    double get_gradient(unsigned layer, unsigned neuron) const
    {
      if(_batch_size == 0 )
      {
        return 0.0;
      }
      return _gradients.get(layer, neuron)/ static_cast<double>(_batch_size);
    }

    void set_gradient(unsigned layer, unsigned neuron, double gradient)
    {
      // if we are calling 'set' then it is for a single batch
      // otherwise call the add_outputs_gradients(...)
      assert(_batch_size== 0 || _batch_size == 1);
      _gradients.set(layer, neuron, gradient);
      _batch_size = 1;
    }

    void set_gradients(unsigned layer, const std::vector<double>& gradients)
    {
      for( unsigned neuron = 0; neuron < gradients.size(); ++neuron)
      {
        set_gradient(layer, neuron, gradients[neuron]);
      }
    }

    void set_gradients(const std::vector<std::vector<double>>& gradients)
    {
      // if we are calling 'set' then it is for a single batch
      // otherwise call the add_outputs_gradients(...)
      assert(_batch_size== 0 || _batch_size == 1);
      _gradients = gradients;
      _batch_size = 1;
    }

    unsigned num_gradient_layers() const 
    { 
      return static_cast<unsigned>(_gradients.number_layers());
    }

    unsigned num_gradient_neurons(unsigned layer) const 
    { 
      return static_cast<unsigned>(_gradients.number_neurons(layer));
    }

    double get_output(unsigned layer, unsigned neuron) const
    {
      if(_outputs.number_neurons(layer) == neuron)
      {
        return 1.0; //  bias
      }      
      return _outputs.get(layer, neuron);
    }

    std::vector<double> get_outputs(unsigned layer) const
    {
      if(_outputs.number_layers() <= layer)
      {
        //  just the bias
        return {1.0};
      }
      //  add the bias
      auto outputs = _outputs.get_neurons(layer);
      outputs.push_back(1.0);
      return outputs;
    }

    unsigned num_outputs(unsigned layer) const
    {
      //  add the bias
      return static_cast<unsigned>(_outputs.number_neurons(layer) + 1);
    }

    void set_output(unsigned layer, unsigned neuron, double output)
    {
      _outputs.set(layer, neuron, output);
    }
    
    void set_outputs(unsigned layer, const std::vector<double>& outputs)
    {
      for(unsigned neuron = 0; neuron < outputs.size(); ++neuron)
      {
        set_output(layer, neuron, outputs[neuron]);
      }
    }

    void zero()
    {
      _gradients.zero();
      _outputs.zero();
      zero_outputs_gradients();
    }
    const std::vector<double> output_layer_outputs(bool include_bias) const
    {
      const size_t size = _outputs.number_layers();
      if(size == 0)
      {
        return {};
      }
      auto output = _outputs.get_neurons(size -1);
      if(true == include_bias)
      {
        output.push_back(1.0);
      }
      return output;
    }

  private:
    LayersAndNeurons _outputs;
    LayersAndNeurons _gradients;
  };

public:
  NeuralNetwork(const std::vector<unsigned>& topology, const activation::method& activation, double learning_rate);
  NeuralNetwork(const std::vector<Layer>& layers, const activation::method& activation, double learning_rate, double error);
  NeuralNetwork(const NeuralNetwork& src);
  NeuralNetwork& operator=(const NeuralNetwork&) = delete;

  virtual ~NeuralNetwork();

  void train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, int number_of_epoch, int batch_size = 1, bool data_is_unique = true, const std::function<bool(int, NeuralNetwork&)>& progress_callback = nullptr);

  std::vector<std::vector<double>> think(
    const std::vector<std::vector<double>>& inputs
  ) const;
  std::vector<double> think(
    const std::vector<double>& inputs
  ) const;

  std::vector<unsigned> get_topology() const;
  const std::vector<Layer>& get_layers() const;
  activation::method get_activation_method() const;
  long double get_error() const;
  double get_learning_rate() const;

private:
  static GradientsAndOutputs calculate_forward_feed(const std::vector<double>& inputs, const std::vector<Layer>& layers);
  static std::vector<GradientsAndOutputs> calculate_forward_feed(const std::vector<std::vector<double>>& inputs, const std::vector<Layer>& layers);
  
  static GradientsAndOutputs average_batch_gradients(const std::vector<GradientsAndOutputs>& batch_activation_gradients);
  static void calculate_batch_back_propagation(const std::vector<std::vector<double>>& target_outputs, std::vector<GradientsAndOutputs>& batch_given_outputs, const std::vector<Layer>& layers);

  static void update_layers_with_gradients(const AverageGradientsAndOutputs& activation_gradients, std::vector<Layer>& layers);
  static void update_layers_with_gradients(const std::vector<std::vector<GradientsAndOutputs>>& batch_activation_gradients, std::vector<Layer>& layers);

  static GradientsAndOutputs average_batch_gradients_with_averages(const GradientsAndOutputs& activation_gradients, const std::vector<std::vector<double>>& averages);
  static GradientsAndOutputs average_batch_gradients_with_averages(const std::vector<GradientsAndOutputs>& batch_activation_gradients, const std::vector<std::vector<double>>& averages);
  static std::vector<std::vector<double>> recalculate_gradient_avergages(const std::vector<std::vector<GradientsAndOutputs>>& epoch_gradients_outputs);
  
  static void calculate_back_propagation_gradients(const std::vector<double>& target_outputs, GradientsAndOutputs& layers_given_outputs, const std::vector<Layer>& layers);
  static void calculate_batch_back_propagation_gradients(const std::vector<std::vector<double>>& target_outputs, std::vector<GradientsAndOutputs>& layers_given_outputs, const std::vector<Layer>& layers);

  static std::vector<double> caclulate_output_gradients(const std::vector<double>& target_outputs, const std::vector<double>& given_outputs, const Layer& output_layer);

  // Todo this should be moved to a static class a passed as an object.
  static double calculate_error(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<Layer>& layers);

  // Huber Loss blends MAE and RMSE — it uses squared error when the difference is small (|error| < delta), and absolute error when it’s large.
  static double calculate_huber_loss(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions, double delta = 1.0);

  // MAE is more robust to outliers than RMSE.
  static double calculate_mae_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions);
  static double calculate_mse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions);
  static double calculate_rmse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions );

  static void create_batch_from_indexes(const std::vector<size_t>& shuffled_indexes, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& shuffled_training_inputs, std::vector<std::vector<double>>& shuffled_training_outputs);
  static void break_shuffled_indexes(const std::vector<size_t>& shuffled_indexes, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes);
  static void create_shuffled_indexes(size_t raw_size, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes);
  static std::vector<size_t> get_shuffled_indexes(size_t raw_size);
  
  long double _error;
  std::vector<Layer>* _layers;
  const activation::method _activation_method;
  double _learning_rate;
};