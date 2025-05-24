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
  class GradientsAndOutputs
  {
    public:
      GradientsAndOutputs(){};
      GradientsAndOutputs(const GradientsAndOutputs& src) :
        _gradients(src._gradients),
        _outputs(src._outputs),
        _outputs_gradients(src._outputs_gradients)
      {
      };
      GradientsAndOutputs& operator=(const GradientsAndOutputs& src)
      {
        if( &src != this)
        {
          _gradients = src._gradients;
          _outputs = src._outputs;
          _outputs_gradients = src._outputs_gradients;
        }
        return *this;
      }
      virtual ~GradientsAndOutputs() = default;

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
        if(_gradients.size() <= layer)
        {
          return 0;
        }
        if(_gradients[layer].size() <= neuron)
        {
          return 0;
        }
        return _gradients[layer][neuron];
      }

      void set_gradient(unsigned layer, unsigned neuron, double gradient)
      {
        while(_gradients.size() <= layer)
        {
          _gradients.push_back({});
        }
        while(_gradients[layer].size() <= neuron)
        {
          _gradients[layer].push_back(0.0);
        }
        _gradients[layer][neuron] = gradient;
      }

      void set_gradients(unsigned layer, const std::vector<double>& gradients)
      {
        while(_gradients.size() <= layer)
        {
          _gradients.push_back({});
        }
        _gradients[layer] = gradients;
      }

      double get_output(unsigned layer, unsigned neuron) const
      {
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

      void update_outputs_gradients()
      {
        zero_outputs_gradients();
        size_t number_output_layers = _outputs.size();
        size_t number_gradient_layers = _gradients.size();
        assert(number_gradient_layers == number_output_layers);

        _outputs_gradients.resize(number_output_layers-1);
        for( size_t layer = 0; layer < number_output_layers -1; ++layer ) // we always get the next layer target neurons
        {
          size_t number_neurons = _outputs[layer].size();
          _outputs_gradients[layer].resize(number_neurons);
          for( size_t target_neuron = 0; target_neuron < number_neurons; ++target_neuron )
          {
            // get the gradient this neuron in the next layer
            const auto& next_layer_neuron_gradient = _gradients[layer+1][target_neuron];

            // get the output[layer] * gradient[layer+1]
            std::vector<double> outputs_gradients;
            size_t number_outputs = _outputs[layer].size();
            outputs_gradients.resize(number_outputs +1, 0.0);
            for (size_t output_number = 0; output_number < number_outputs; ++output_number)
            {
              const auto& layer_neuron_output = _outputs[layer][output_number];
              outputs_gradients[output_number] = layer_neuron_output * next_layer_neuron_gradient;
            }
            
            // bias
            outputs_gradients[number_outputs] = 1 * next_layer_neuron_gradient;
            set_outputs_gradients(layer, target_neuron, outputs_gradients);
          }
        }
      }

      void set_outputs_gradients(unsigned layer, unsigned target_neuron, const std::vector<double>& outputs_gradients)
      {
        while(_outputs_gradients.size() <= layer)
        {
          _outputs_gradients.push_back({});
        }
        while(_outputs_gradients[layer].size() <= target_neuron)
        {
          _outputs_gradients[layer].push_back({});
        }
        _outputs_gradients[layer][target_neuron] = outputs_gradients;
      }

      std::vector<double> get_outputs_gradients(unsigned layer, unsigned target_neuron) const
      {
        if(_outputs_gradients.size() <= layer)
        {
          return {};
        }
        if(_outputs_gradients[layer].size() <= target_neuron)
        {
          return {};
        }
        return _outputs_gradients[layer][target_neuron];
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
        while(_outputs.size() <= layer)
        {
          _outputs.push_back({});
        }
        _outputs[layer] = outputs;
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
      void zero_gradients(){ _gradients = {};};
      void zero_outputs(){ _outputs = {};};
      void zero_outputs_gradients(){ _outputs_gradients = {};};

      std::vector<std::vector<double>> _gradients;
      std::vector<std::vector<double>> _outputs;
      std::vector<std::vector<std::vector<double>>> _outputs_gradients;
  };
public:
  NeuralNetwork(const std::vector<unsigned>& topology, const activation::method& activation, double learning_rate);
  NeuralNetwork(const std::vector<Layer>& layers, const activation::method& activation, double learning_rate, double error);
  NeuralNetwork(const NeuralNetwork& src);
  NeuralNetwork& operator=(const NeuralNetwork&) = delete;

  virtual ~NeuralNetwork();

  void train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, int number_of_epoch, int batch_size = -1, bool data_is_unique = true, const std::function<bool(int, NeuralNetwork&)>& progress_callback = nullptr);

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
  void train( const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, int number_of_epoch, bool data_is_unique, const std::function<bool(int, NeuralNetwork&)>& progress_callback);
  void train_in_batch( const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, int number_of_epoch, int batch_size, bool data_is_unique, const std::function<bool(int, NeuralNetwork&)>& progress_callback);

  static GradientsAndOutputs calculate_forward_feed(const std::vector<double>& inputs, const std::vector<Layer>& layers);
  static std::vector<GradientsAndOutputs> calculate_forward_feed(const std::vector<std::vector<double>>& inputs, const std::vector<Layer>& layers);
  
  static GradientsAndOutputs average_batch_gradients(const std::vector<GradientsAndOutputs>& batch_activation_gradients);
  static void batch_back_propagation(const std::vector<std::vector<double>>& target_outputs, std::vector<GradientsAndOutputs>& batch_given_outputs, std::vector<Layer>& layers);
  static void back_propagation(const std::vector<double>& target_outputs, GradientsAndOutputs& given_outputs, std::vector<Layer>& layers);
  static void update_layers_with_gradients(GradientsAndOutputs& activation_gradients, std::vector<Layer>& layers);
  
  static void calculate_back_propagation_gradients(const std::vector<double>& target_outputs, GradientsAndOutputs& layers_given_outputs, const std::vector<Layer>& layers);
  static void calculate_batch_back_propagation_gradients(const std::vector<std::vector<double>>& target_outputs, std::vector<GradientsAndOutputs>& layers_given_outputs, const std::vector<Layer>& layers);

  static void set_output_gradients(const std::vector<double>& target_outputs, Layer& output_layer);
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