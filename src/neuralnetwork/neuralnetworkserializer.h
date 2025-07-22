#pragma once
#include <array>
#include <string>
#include <vector>

#include "layer.h"
#include "logger.h"
#include "neuralnetwork.h"
#include "neuron.h"
#include "weightparam.h"
#include "libraries/TinyJSON.h"

class NeuralNetworkSerializer
{
public:
  static NeuralNetwork* load(Logger& logger, const std::string& path);
  static void save(const NeuralNetwork& nn, const std::string& path);

private:
  NeuralNetworkSerializer();
  NeuralNetworkSerializer( const NeuralNetworkSerializer& src) = delete;
  NeuralNetworkSerializer& operator=(const NeuralNetworkSerializer& src) = delete;
  virtual ~NeuralNetworkSerializer() = default;

  static NeuralNetworkOptions get_options(Logger& logger, const TinyJSON::TJValue& json);
  static double get_error(const TinyJSON::TJValue& json);
  static double get_mean_absolute_percentage_error(const TinyJSON::TJValue& json);
  static std::vector<WeightParam> get_weight_params(Logger& logger, const TinyJSON::TJValueObject& neuron);
  static std::vector<Neuron> get_neurons(Logger& logger, const TinyJSON::TJValue& json, unsigned layer_number,const activation::method& activation_method);
  static std::vector<int> get_residual_layers(Logger& logger, const TinyJSON::TJValue& json);
  static std::vector<Layer> create_layers(Logger& logger, std::vector<std::vector<Neuron>> array_of_neurons, const std::vector<int>& residual_layers);

  static void add_basic(TinyJSON::TJValueObject& json);
  static void add_errors(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_layers(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_layer(const Layer& layer, TinyJSON::TJValueArray& layers);
  static TinyJSON::TJValueObject* add_neuron(const Neuron& neuron);
  static void add_weight_params(const std::vector<WeightParam>& weight_params, TinyJSON::TJValueObject& neuron);
  static void add_options(const NeuralNetworkOptions& options, TinyJSON::TJValueObject& json);
  static void add_final_learning_rate(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
};