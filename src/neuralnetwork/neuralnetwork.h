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
    LayersAndNeurons(const std::vector<unsigned>& topology) noexcept
    {
      _data.reserve(topology.size());
      for(size_t layer = 0; layer < topology.size(); ++layer)
      {
        _data.emplace_back(topology[layer]);
      }
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

    const std::vector<double>& get_neurons(size_t layer) const 
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

    void set(size_t layer, const std::vector<double> values) 
    {
      ensure_size(layer, values.size()-1);
      _data[layer] = values;
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

  class FlatAverageGradientsAndOutputs
  {
  public:
    FlatAverageGradientsAndOutputs(const FlatAverageGradientsAndOutputs& src) noexcept :
      _batch_size(src._batch_size),
      _data(src._data),
      _offsets(src._offsets)
    {
    }

    FlatAverageGradientsAndOutputs(FlatAverageGradientsAndOutputs&& src) noexcept :
      _batch_size(src._batch_size),
      _data(std::move(src._data)),
      _offsets(std::move(src._offsets))
    {
      src._batch_size = 0;
    }

    FlatAverageGradientsAndOutputs& operator=(const FlatAverageGradientsAndOutputs& src) noexcept
    {
      if(this != &src)
      {
        _batch_size = src._batch_size;
        _data = src._data;
        _offsets = src._offsets;
      }
      return *this;
    }

    FlatAverageGradientsAndOutputs& operator=(FlatAverageGradientsAndOutputs&& src) noexcept
    {
      if(this != &src)
      {
        _batch_size = src._batch_size;
        _data = std::move(src._data);
        _offsets = std::move(src._offsets);
        src._batch_size = 0;
      }
      return *this;
    }

    FlatAverageGradientsAndOutputs(const std::vector<unsigned>& topology) noexcept :
      _batch_size(0)
    {
      size_t offset = 0;
      // the offset size is the nuber of layers.
      _offsets.reserve(topology.size() -1);
      for (size_t layer = 1; layer < topology.size(); ++layer)
      {
        _offsets.emplace_back(topology[layer]+1);
        for (size_t neuron = 0; neuron < topology[layer]; ++neuron) 
        {
          _offsets[layer-1][neuron] = offset;
          ++offset;
        }
        _offsets[layer-1][topology[layer]] = offset;  //  bias
        ++offset;
      }
      _data.resize(offset);
    }

    inline void add( const FlatAverageGradientsAndOutputs& src)
    {
      for(size_t layer = 0; layer < src._data.size(); ++layer)
      {
        for(size_t neuron = 0; neuron < src._data[layer].size(); ++neuron)
        {
          ensure_size(layer, neuron);
          for( size_t raw_data = 0; raw_data < src._data[_offsets[layer][neuron]].size(); ++raw_data )
          {
            if(_data[_offsets[layer][neuron]].size() <= raw_data)
            {
              _data[_offsets[layer][neuron]].push_back(src._data[src._offsets[layer][neuron]][raw_data]);
            }
            else
            {
              _data[_offsets[layer][neuron]][raw_data] += src._data[src._offsets[layer][neuron]][raw_data];
            }
          }
        }
      }
      ++_batch_size;
    }

    inline void set( unsigned layer, unsigned neuron, const std::vector<double>&& data)
    {
      ensure_size(layer, neuron);
      _data[_offsets[layer][neuron]] = std::move(data);
      _batch_size = 1;
    }

    inline void set( unsigned layer, unsigned neuron, const std::vector<double>& data)
    {
      ensure_size(layer, neuron);
      _data[_offsets[layer][neuron]] = data;
      _batch_size = 1;
    }

    inline const std::vector<double>& get( unsigned layer, unsigned neuron) const
    {
      ensure_size(layer, neuron);
      return _data[_offsets[layer][neuron]];
    }

    inline int get_batch_size() const 
    {
      return _batch_size;
    }
    inline void set_batch_size(int batch_size)
    {
      _batch_size = batch_size;
    }
  private:
    void ensure_size(size_t layer, size_t neuron) const
    {
      if(layer >= _offsets.size() || neuron >= _offsets[layer].size())
      {
        std::cerr << "The layer/neuron is out of bound!" << std::endl;
        throw new std::invalid_argument("The layer/neuron is out of bound!");
      }
    }

    // the values
    int _batch_size = 0;
    std::vector<std::vector<double>> _data;
    std::vector<std::vector<size_t>> _offsets;
  };

  class GradientsAndOutputs
  {
  public:
    GradientsAndOutputs() = delete;

    GradientsAndOutputs(const std::vector<unsigned>& topology) noexcept: 
      _gradients(topology),
      _gradients_and_outputs(topology)
    {
    }

    GradientsAndOutputs(const GradientsAndOutputs& src) noexcept:
      _outputs(src._outputs),
      _gradients(src._gradients),
      _gradients_and_outputs(src._gradients_and_outputs)
    {
    };
    GradientsAndOutputs(GradientsAndOutputs&& src) noexcept: 
      _outputs(std::move(src._outputs)),
      _gradients(std::move(src._gradients)),
      _gradients_and_outputs(std::move(src._gradients_and_outputs))
    {
    };
    GradientsAndOutputs& operator=(const GradientsAndOutputs& src) noexcept
    {
      if( &src != this)
      {
        _outputs = src._outputs;
        _gradients = src._gradients;
        _gradients_and_outputs = src._gradients_and_outputs;
      }
      return *this;
    }
    GradientsAndOutputs& operator=(GradientsAndOutputs&& src) noexcept
    {
      if( &src != this)
      {
        _outputs = std::move(src._outputs);
        _gradients = std::move(src._gradients);
        _gradients_and_outputs = std::move(src._gradients_and_outputs);
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

    inline const FlatAverageGradientsAndOutputs& get_gradients_and_outputs() const
    {
      return _gradients_and_outputs;
    } 

    inline void set_gradients_and_outputs(unsigned layer, unsigned neuron, const std::vector<double>& data)
    {
      return _gradients_and_outputs.set(layer, neuron, data);
    } 

    double get_gradient(unsigned layer, unsigned neuron) const
    {
      auto batch_size = _gradients_and_outputs.get_batch_size();
      if(batch_size == 0 )
      {
        return 0.0;
      }
      return _gradients.get(layer, neuron)/ static_cast<double>(batch_size);
    }

    void set_gradient(unsigned layer, unsigned neuron, double gradient)
    {
      // if we are calling 'set' then it is for a single batch
      // otherwise call the add_outputs_gradients(...)
      assert(_gradients_and_outputs.get_batch_size()== 0 || _gradients_and_outputs.get_batch_size()==1);
      _gradients.set(layer, neuron, gradient);
      _gradients_and_outputs.set_batch_size(1);
    }

    void set_gradients(unsigned layer, const std::vector<double>& gradients)
    {
      _gradients.set(layer, gradients);
      _gradients_and_outputs.set_batch_size(1);
    }

    void set_gradients(const std::vector<std::vector<double>>& gradients)
    {
      // if we are calling 'set' then it is for a single batch
      // otherwise call the add_outputs_gradients(...)
      assert(_gradients_and_outputs.get_batch_size()== 0 || _gradients_and_outputs.get_batch_size()==1);
      _gradients = gradients;
      _gradients_and_outputs.set_batch_size(1);
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

    unsigned num_outputs(unsigned layer) const
    {
      //  add the bias
      return static_cast<unsigned>(_outputs.number_neurons(layer) + 1);
    }
    
    void set_outputs(unsigned layer, const std::vector<double>& outputs)
    {
      _outputs.set(layer, outputs);
    }

    const std::vector<double>& output_back() const
    {
      const size_t size = _outputs.number_layers();
      if(size == 0)
      {
        std::cerr << "Trying to get the last output but none available!" << std::endl;
        throw new std::invalid_argument("Trying to get the last output but none available!");
      }
      return _outputs.get_neurons(size -1);
    }

  private:
    LayersAndNeurons _outputs;
    LayersAndNeurons _gradients;
    FlatAverageGradientsAndOutputs _gradients_and_outputs;
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

  const std::vector<unsigned>& get_topology() const;
  const std::vector<Layer>& get_layers() const;
  activation::method get_activation_method() const;
  long double get_error() const;
  double get_learning_rate() const;

private:
  std::vector<GradientsAndOutputs> train_single_batch(const std::vector<std::vector<double>>& batch_inputs, const std::vector<std::vector<double>>& batch_outputs) const;

  GradientsAndOutputs calculate_forward_feed(const std::vector<double>& inputs, const std::vector<Layer>& layers) const;
  std::vector<GradientsAndOutputs> calculate_forward_feed(const std::vector<std::vector<double>>& inputs, const std::vector<Layer>& layers) const;
  
  GradientsAndOutputs average_batch_gradients(const std::vector<GradientsAndOutputs>& batch_activation_gradients) const;
  static void calculate_batch_back_propagation(const std::vector<std::vector<double>>& target_outputs, std::vector<GradientsAndOutputs>& batch_given_outputs, const std::vector<Layer>& layers);

  void update_layers_with_gradients(const FlatAverageGradientsAndOutputs& activation_gradients, std::vector<Layer>& layers) const;
  void update_layers_with_gradients(const std::vector<std::vector<GradientsAndOutputs>>& batch_activation_gradients, std::vector<Layer>& layers) const;

  GradientsAndOutputs average_batch_gradients_with_averages(const GradientsAndOutputs& activation_gradients, const std::vector<std::vector<double>>& averages) const;
  GradientsAndOutputs average_batch_gradients_with_averages(const std::vector<GradientsAndOutputs>& batch_activation_gradients, const std::vector<std::vector<double>>& averages) const;
  static std::vector<std::vector<double>> recalculate_gradient_avergages(const std::vector<std::vector<GradientsAndOutputs>>& epoch_gradients_outputs);
  
  static void calculate_back_propagation_gradients(const std::vector<double>& target_outputs, GradientsAndOutputs& layers_given_outputs, const std::vector<Layer>& layers);
  static void calculate_batch_back_propagation_gradients(const std::vector<std::vector<double>>& target_outputs, std::vector<GradientsAndOutputs>& layers_given_outputs, const std::vector<Layer>& layers);

  static std::vector<double> caclulate_output_gradients(const std::vector<double>& target_outputs, const std::vector<double>& given_outputs, const Layer& output_layer);

  // Todo this should be moved to a static class a passed as an object.
  double calculate_error(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<Layer>& layers) const;

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
  std::vector<unsigned> _topology;
  std::vector<Layer>* _layers;
  const activation::method _activation_method;
  double _learning_rate;
};