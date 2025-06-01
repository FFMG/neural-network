#pragma once
#include <cassert>
#include <functional>
#include <vector>

#include "activation.h"
#include "layer.h"
#include "neuron.h"
#include "threadpool.h"

class NeuralNetwork
{
private:
  template <typename T>
  class LayersAndNeurons
  {
  public:
    LayersAndNeurons(const LayersAndNeurons& src) noexcept :
      _data(src._data),
      _offsets(src._offsets)
    {
    }

    LayersAndNeurons(LayersAndNeurons&& src) noexcept :
      _data(std::move(src._data)),
      _offsets(std::move(src._offsets))
    {
    }

    LayersAndNeurons& operator=(const LayersAndNeurons& src) noexcept
    {
      if(this != &src)
      {
        _data = src._data;
        _offsets = src._offsets;
      }
      return *this;
    }

    LayersAndNeurons& operator=(LayersAndNeurons&& src) noexcept
    {
      if(this != &src)
      {
        _data = std::move(src._data);
        _offsets = std::move(src._offsets);
      }
      return *this;
    }

    LayersAndNeurons(const std::vector<unsigned>& topology, bool shifted_by_one=false, bool add_bias=false) noexcept
    {
      size_t offset = 0;
      if(shifted_by_one)
      {
        _offsets.reserve(topology.size() -1);
        for (size_t layer = 1; layer < topology.size(); ++layer)
        {
          _offsets.emplace_back(topology[layer]+(add_bias?1:0));
          for (size_t neuron = 0; neuron < topology[layer]; ++neuron) 
          {
            _offsets[layer-1][neuron] = offset;
            ++offset;
          }
          if(add_bias)
          {
            _offsets[layer-1][topology[layer]] = offset;  //  bias
            ++offset;
          }
        }
      }
      else
      {
        _offsets.reserve(topology.size());
        for (size_t layer = 0; layer < topology.size(); ++layer)
        {
          _offsets.emplace_back(topology[layer]+(add_bias?1:0));
          for (size_t neuron = 0; neuron < topology[layer]; ++neuron) 
          {
            _offsets[layer][neuron] = offset;
            ++offset;
          }
          if(add_bias)
          {
            _offsets[layer][topology[layer]] = offset;  //  bias
            ++offset;
          }          
        }
      }
      _data.resize(offset);
    }

    LayersAndNeurons& operator=(const std::vector<std::vector<T>>& data)
    {
      assert(_offsets.size() == data.size());
      for(size_t layer = 0; layer < data.size(); ++layer)
      {
        set(layer, data[layer]);
      }
      return *this;
    }

    inline void set( unsigned layer, unsigned neuron, const T&& data)
    {
      ensure_size(layer, neuron);
      _data[_offsets[layer][neuron]] = std::move(data);
    }

    inline void set( unsigned layer, unsigned neuron, const T& data)
    {
      ensure_size(layer, neuron);
      _data[_offsets[layer][neuron]] = data;
    }

    inline void set( unsigned layer, const std::vector<T>& data)
    {
      assert(number_neurons(layer) == data.size());
      for(size_t neuron = 0; neuron < data.size(); ++neuron)
      {
        _data[_offsets[layer][neuron]] = data[neuron];
      }
    }
    
    inline const T& get( unsigned layer, unsigned neuron) const
    {
      ensure_size(layer, neuron);
      return _data[_offsets[layer][neuron]];
    }

    std::vector<T> get_neurons( unsigned layer) const
    {
      std::vector<T> data;
      data.reserve(_offsets[layer].size());
      for(size_t neuron = 0; neuron < _offsets[layer].size(); ++neuron)
      {
        data.emplace_back(_data[_offsets[layer][neuron]]);
      }
      return data;
    }

    size_t number_layers() const
    {
      return _offsets.size();
    }

    size_t number_neurons(size_t layer) const
    {
      if(layer >= _offsets.size())
      {
        return 0;
      }
      // if we have a topology of  {1,2,6}
      // then the number of neurons at layer 1 = 2
      // the offset are
      // layer[0][0] = 0
      // layer[1][0] = 1
      // layer[1][1] = 2
      // layer[2][0] = 3
      // layer[2][1] = 4
      // ...
      // so the number of neurons is the number of offsets at that layer
      return _offsets[layer].size();
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
    std::vector<T> _data;
    std::vector<std::vector<size_t>> _offsets;
  };

  class GradientsAndOutputs
  {
  public:
    GradientsAndOutputs() = delete;

    GradientsAndOutputs(const std::vector<unsigned>& topology) noexcept: 
      _batch_size(0),
      _outputs(topology, false, false),
      _gradients(topology, false, true),
      _gradients_and_outputs(topology, true, true)
    {
    }

    GradientsAndOutputs(const GradientsAndOutputs& src) noexcept:
      _batch_size(src._batch_size),
      _outputs(src._outputs),
      _gradients(src._gradients),
      _gradients_and_outputs(src._gradients_and_outputs)
    {
    };
    GradientsAndOutputs(GradientsAndOutputs&& src) noexcept: 
      _batch_size(src._batch_size),
      _outputs(std::move(src._outputs)),
      _gradients(std::move(src._gradients)),
      _gradients_and_outputs(std::move(src._gradients_and_outputs))
    {
      src._batch_size = 0;
    }

    GradientsAndOutputs& operator=(const GradientsAndOutputs& src) noexcept
    {
      if( &src != this)
      {
        _batch_size = src._batch_size;
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
        _batch_size = src._batch_size;
        _outputs = std::move(src._outputs);
        _gradients = std::move(src._gradients);
        _gradients_and_outputs = std::move(src._gradients_and_outputs);
        src._batch_size = 0;
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

    inline const LayersAndNeurons<std::vector<double>>& get_gradients_and_outputs() const
    {
      return _gradients_and_outputs;
    } 

    inline void set_gradients_and_outputs(unsigned layer, unsigned neuron, const std::vector<double>& data)
    {
      return _gradients_and_outputs.set(layer, neuron, data);
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
      assert(_batch_size == 0 || _batch_size == 1);
      _gradients.set(layer, neuron, gradient);
      _batch_size = 1;
    }

    void set_gradients(unsigned layer, const std::vector<double>& gradients)
    {
      _gradients.set(layer, gradients);
      _batch_size = 1;
    }

    void set_gradients(const LayersAndNeurons<double>& gradients)
    {
      _gradients = gradients;
      _batch_size = 1;
    }

    void set_gradients(const std::vector<std::vector<double>>& gradients)
    {
      // if we are calling 'set' then it is for a single batch
      // otherwise call the add_outputs_gradients(...)
      assert(_batch_size == 0 || _batch_size == 1);
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

    unsigned num_outputs(unsigned layer) const
    {
      //  add the bias
      return static_cast<unsigned>(_outputs.number_neurons(layer) + 1);
    }
    
    void set_outputs(unsigned layer, const std::vector<double>& outputs)
    {
      _outputs.set(layer, outputs);
    }

    std::vector<double> output_back() const
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
    int _batch_size;
    LayersAndNeurons<double> _outputs;
    LayersAndNeurons<double> _gradients;
    LayersAndNeurons<std::vector<double>> _gradients_and_outputs;
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

  std::vector<GradientsAndOutputs> train_single_batch(
    const std::vector<std::vector<double>>::const_iterator inputs_begin, 
    const std::vector<std::vector<double>>::const_iterator outputs_begin,
    const size_t size
  ) const;
  std::vector<GradientsAndOutputs> calculate_forward_feed(
    const std::vector<std::vector<double>>::const_iterator inputs_begin, 
    const std::vector<std::vector<double>>::const_iterator inputs_end, 
    const std::vector<Layer>& layers) const;
  static void calculate_batch_back_propagation(
    const std::vector<std::vector<double>>::const_iterator outputs_begin, 
    const size_t outputs_size, 
    std::vector<GradientsAndOutputs>& batch_given_outputs, 
    const std::vector<Layer>& layers);
  static void calculate_batch_back_propagation_gradients(
    const std::vector<std::vector<double>>::const_iterator outputs_begin, 
    const size_t outputs_size, 
    std::vector<GradientsAndOutputs>& layers_given_outputs, 
    const std::vector<Layer>& layers);
  
  GradientsAndOutputs average_batch_gradients(const std::vector<GradientsAndOutputs>& batch_activation_gradients) const;
  static void calculate_batch_back_propagation(const std::vector<std::vector<double>>& target_outputs, std::vector<GradientsAndOutputs>& batch_given_outputs, const std::vector<Layer>& layers);

  void update_layers_with_gradients(const LayersAndNeurons<std::vector<double>>& activation_gradients, std::vector<Layer>& layers) const;
  void update_layers_with_gradients(const std::vector<std::vector<GradientsAndOutputs>>& batch_activation_gradients, std::vector<Layer>& layers) const;

  GradientsAndOutputs average_batch_gradients_with_averages(const GradientsAndOutputs& activation_gradients, const LayersAndNeurons<double>& averages) const;
  LayersAndNeurons<double> recalculate_gradient_avergages(const std::vector<std::vector<GradientsAndOutputs>>& epoch_gradients_outputs) const;
  
  static void calculate_back_propagation_gradients(const std::vector<double>& target_outputs, GradientsAndOutputs& layers_given_outputs, const std::vector<Layer>& layers);
  static void calculate_batch_back_propagation_gradients(const std::vector<std::vector<double>>& target_outputs, std::vector<GradientsAndOutputs>& layers_given_outputs, const std::vector<Layer>& layers);

  static std::vector<double> caclulate_output_gradients(const std::vector<double>& target_outputs, const std::vector<double>& given_outputs, const Layer& output_layer);

  // Todo this should be moved to a static class a passed as an object.
  double calculate_error(ThreadPool& threadpool, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<Layer>& layers) const;

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