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
  class AverageGradientsAndOutputs
  {
  public:
    AverageGradientsAndOutputs() :
      _batch_size(0)
    {

    };
    AverageGradientsAndOutputs(const AverageGradientsAndOutputs& src) :
      _outputs_gradients(src._outputs_gradients),
      _batch_size(src._batch_size)
    {
    }
    AverageGradientsAndOutputs& operator=(const AverageGradientsAndOutputs& src)
    {
      if( &src != this)
      {
        _outputs_gradients = src._outputs_gradients;
        _batch_size = src._batch_size;
      }
      return *this;
    }

    void add_outputs_gradients(const AverageGradientsAndOutputs& average_outputs_gradients)
    {
      const auto& outputs_gradients = average_outputs_gradients.get_outputs_gradients();
      for(size_t layer = 0; layer < outputs_gradients.size(); ++layer)
      {
        while(_outputs_gradients.size() <= layer)
        {
          _outputs_gradients.push_back({});
        }
        for(size_t neuron = 0; neuron < outputs_gradients[layer].size(); ++neuron)
        {
          while(_outputs_gradients[layer].size() <= neuron)
          {
            _outputs_gradients[layer].push_back({});
          }

          size_t number_outputs = outputs_gradients[layer][neuron].size();
          if(_outputs_gradients[layer][neuron].size() == 0 )
          {
            _outputs_gradients[layer][neuron].resize(number_outputs, 0.0);
          }

          for(size_t i = 0; i < number_outputs; ++i)
          {
            _outputs_gradients[layer][neuron][i] += outputs_gradients[layer][neuron][i];
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
      while(_outputs_gradients.size() <= layer)
      {
        _outputs_gradients.push_back({});
      }
      while(_outputs_gradients[layer].size() <= target_neuron)
      {
        _outputs_gradients[layer].push_back({});
      }
      _outputs_gradients[layer][target_neuron] = outputs_gradients;
      _batch_size = 1;
    }

    std::vector<double> get_outputs_gradients(unsigned layer, unsigned target_neuron) const
    {
      if(_batch_size == 0 )
      {
        return {};
      }
      if(_outputs_gradients.size() <= layer)
      {
        return {};
      }
      if(_outputs_gradients[layer].size() <= target_neuron)
      {
        return {};
      }
      if(_batch_size == 1)
      {
        return _outputs_gradients[layer][target_neuron];
      }
      std::vector<double> average_output_gradient = {};
      for( const auto& output : _outputs_gradients[layer][target_neuron])
      {
        average_output_gradient.push_back(output / static_cast<double>(_batch_size));
      }
      return average_output_gradient;
    }

  protected:
    void zero_outputs_gradients(){ _outputs_gradients = {};};

    const std::vector<std::vector<std::vector<double>>>& get_outputs_gradients() const
    {
      return _outputs_gradients;
    }

    std::vector<std::vector<std::vector<double>>> _outputs_gradients;
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
    GradientsAndOutputs& operator=(const GradientsAndOutputs& src)
    {
      if( &src != this)
      {
        AverageGradientsAndOutputs::operator=(src);
        _outputs = src._outputs;
        _gradients = src._gradients;
      }
      return *this;
    }
    virtual ~GradientsAndOutputs() = default;

    unsigned num_output_layers() const 
    { 
      return static_cast<unsigned>(_outputs.size());
    }

    unsigned num_output_neurons(unsigned layer) const 
    { 
      if(_outputs.size() <= layer)
      {
        return 0;
      }
      return static_cast<unsigned>(_outputs[layer].size());
    }

    double get_gradient(unsigned layer, unsigned neuron) const
    {
      if(_batch_size == 0 )
      {
        return 0.0;
      }
      if(_gradients.size() <= layer)
      {
        return 0.0;
      }
      if(_gradients[layer].size() <= neuron)
      {
        return 0.0;
      }
      return _gradients[layer][neuron] / static_cast<double>(_batch_size);
    }

    void set_gradient(unsigned layer, unsigned neuron, double gradient)
    {
      // if we are calling 'set' then it is for a single batch
      // otherwise call the add_outputs_gradients(...)
      assert(_batch_size== 0 || _batch_size == 1);
      while(_gradients.size() <= layer)
      {
        _gradients.push_back({});
      }
      while(_gradients[layer].size() <= neuron)
      {
        _gradients[layer].push_back(0.0);
      }
      _gradients[layer][neuron] = gradient;
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
      return static_cast<unsigned>(_gradients.size());
    }

    unsigned num_gradient_neurons(unsigned layer) const 
    { 
      if(_gradients.size() <= layer)
      {
        return 0;
      }
      return static_cast<unsigned>(_gradients[layer].size());
    }

    double get_output(unsigned layer, unsigned neuron) const
    {
      if(_outputs[layer].size()+1 == neuron)
      {
        return 1; //  bias
      }
      if(_outputs.size() <= layer)
      {
        return 0;
      }
      if(_outputs[layer].size() <= neuron)
      {
        return 0;
      }
      return _outputs[layer][neuron];
    }

    std::vector<double> get_outputs(unsigned layer) const
    {
      if(_outputs.size() <= layer)
      {
        //  just the bias
        return {1.0};
      }
      //  add the bias
      auto outputs = _outputs[layer];
      outputs.push_back(1.0);
      return outputs;
    }

    unsigned num_outputs(unsigned layer) const
    {
      if(_outputs.size() <= layer)
      {
        //  just the bias
        return 1;
      }
      //  add the bias
      return static_cast<unsigned>(_outputs[layer].size() + 1);
    }

    void set_output(unsigned layer, unsigned neuron, double output)
    {
      while(_outputs.size() <= layer)
      {
        _outputs.push_back({});
      }
      while(_outputs[layer].size() <= neuron)
      {
        _outputs[layer].push_back(0.0);
      }
      _outputs[layer][neuron] = output;
    }
    
    void set_outputs(unsigned layer, const std::vector<double>& outputs)
    {
      for(unsigned neuron = 0; neuron < outputs.size(); ++neuron)
      {
        set_output(layer, neuron, outputs[neuron]);
      }
    }

    void zero(){
      zero_gradients();
      zero_outputs();
      zero_outputs_gradients();
    }
    const std::vector<double>& back() const{
      return _outputs.back();
    }

  private:
    void zero_outputs(){ _outputs = {};};
    void zero_gradients(){ _gradients = {};};
    std::vector<std::vector<double>> _outputs;
    std::vector<std::vector<double>> _gradients;
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