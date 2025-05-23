#pragma once
#include <vector>
#include <functional>

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
        _activation_gradients(src._activation_gradients),
        _activation_outputs(src._activation_outputs)
      {
      };
      GradientsAndOutputs& operator=(const GradientsAndOutputs& src)
      {
        if( &src != this)
        {
          _activation_gradients = src._activation_gradients;
          _activation_outputs = src._activation_outputs;
        }
        return *this;
      }
      virtual ~GradientsAndOutputs() = default;

      unsigned num_layers() const { return static_cast<unsigned>(_activation_gradients.size());}
      unsigned num_neurons(unsigned layer) const 
      { 
        if(_activation_gradients.size() <= layer)
        {
          return 0;
        }
        return static_cast<unsigned>(_activation_gradients[layer].size());}

      double get_gradient(unsigned layer, unsigned neuron) const
      {
        if(_activation_gradients.size() <= layer)
        {
          return 0;
        }
        if(_activation_gradients[layer].size() <= neuron)
        {
          return 0;
        }
        return _activation_gradients[layer][neuron];
      }
      void set_gradient(unsigned layer, unsigned neuron, double gradient)
      {
        while(_activation_gradients.size() <= layer)
        {
          _activation_gradients.push_back({});
        }
        while(_activation_gradients[layer].size() <= neuron)
        {
          _activation_gradients[layer].push_back(0.0);
        }
        _activation_gradients[layer][neuron] = gradient;
      }

      void set_gradients(unsigned layer, const std::vector<double>& gradients)
      {
        while(_activation_gradients.size() <= layer)
        {
          _activation_gradients.push_back({});
        }
        _activation_gradients[layer] = gradients;
      }

      double get_output(unsigned layer, unsigned neuron) const
      {
        if(_activation_outputs.size() <= layer)
        {
          return 0;
        }
        if(_activation_outputs[layer].size() <= neuron)
        {
          return 0;
        }
        return _activation_outputs[layer][neuron];
      }
      void set_output(unsigned layer, unsigned neuron, double output)
      {
        while(_activation_outputs.size() <= layer)
        {
          _activation_outputs.push_back({});
        }
        while(_activation_outputs[layer].size() <= neuron)
        {
          _activation_outputs[layer].push_back(0.0);
        }
        _activation_outputs[layer][neuron] = output;
      }
      void set_outputs(unsigned layer, const std::vector<double>& outputs)
      {
        while(_activation_outputs.size() <= layer)
        {
          _activation_outputs.push_back({});
        }
        _activation_outputs[layer] = outputs;
      }

      void zero_gradients(){ _activation_gradients = {};};
      void zero_outputs(){ _activation_outputs = {};};

      const std::vector<double>& back() const{
        return _activation_outputs.back();
      }

      std::vector<std::vector<double>> _activation_gradients;
      std::vector<std::vector<double>> _activation_outputs;
  };
public:
  NeuralNetwork(const std::vector<unsigned>& topology, const activation::method& activation, double learning_rate);
  NeuralNetwork(const std::vector<Layer>& layers, const activation::method& activation, double learning_rate, double error);
  NeuralNetwork(const NeuralNetwork& src);
  NeuralNetwork& operator=(const NeuralNetwork&) = delete;

  virtual ~NeuralNetwork();

  void train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, int number_of_epoch, int batch_size = -1, const std::function<bool(int, NeuralNetwork&)>& progress_callback = nullptr);

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
  void train( const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, int number_of_epoch, const std::function<bool(int, NeuralNetwork&)>& progress_callback);
  void train_in_batch( const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, int number_of_epoch, int batch_size, const std::function<bool(int, NeuralNetwork&)>& progress_callback);

  static GradientsAndOutputs calculate_forward_feed(const std::vector<double>& inputs, const std::vector<Layer>& layers);
  static std::vector<GradientsAndOutputs> calculate_forward_feed(const std::vector<std::vector<double>>& inputs, const std::vector<Layer>& layers);

  static std::vector<double> forward_feed(const std::vector<double>& inputs, std::vector<Layer>& layers);
  static std::vector<std::vector<double>> forward_feed(const std::vector<std::vector<double>>& inputs_batch, std::vector<Layer>& layers);
  
  static GradientsAndOutputs average_batch_gradients(const std::vector<GradientsAndOutputs>& batch_activation_gradients);
  static void batch_back_propagation(const std::vector<std::vector<double>>& target_outputs, std::vector<GradientsAndOutputs>& batch_given_outputs, std::vector<Layer>& layers);
  static void back_propagation(const std::vector<double>& target_outputs, GradientsAndOutputs& given_outputs, std::vector<Layer>& layers);
  static void update_layers_with_gradients(const GradientsAndOutputs& activation_gradients, std::vector<Layer>& layers);
  
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
  static void break_shuffled_indexes(const std::vector<size_t>& shuffled_indexes, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes);
  static void create_shuffled_indexes(size_t raw_size, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes);
  static std::vector<size_t> get_shuffled_indexes(size_t raw_size);
  
  long double _error;
  std::vector<Layer>* _layers;
  const activation::method _activation_method;
  double _learning_rate;
};