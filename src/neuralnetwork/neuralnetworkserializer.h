#pragma once
#include <array>
#include <map>
#include <string>
#include <vector>

#include "layer.h"
#include "errorcalculation.h"
#include "neuralnetwork.h"
#include "neuron.h"
#include "weightparam.h"
#include "libraries/TinyJSON.h"

class NeuralNetworkSerializer
{
public:
  static NeuralNetwork* load(const std::string& path);
  static void save(const NeuralNetwork& nn, const std::string& path);

private:
  NeuralNetworkSerializer();
  NeuralNetworkSerializer( const NeuralNetworkSerializer& src) = delete;
  NeuralNetworkSerializer& operator=(const NeuralNetworkSerializer& src) = delete;
  virtual ~NeuralNetworkSerializer() = default;

  static NeuralNetworkOptions get_and_build_options(const TinyJSON::TJValue& json);
  static std::map<ErrorCalculation::type, double> get_errors(const TinyJSON::TJValue& json);
  static double get_mean_absolute_percentage_error(const TinyJSON::TJValue& json);
  static std::vector<WeightParam> get_weight_params(const TinyJSON::TJValueObject& parent);
  static std::vector<Neuron> get_neurons(const TinyJSON::TJValue& json, unsigned layer_number);
  static std::vector<std::vector<WeightParam>> get_residual_weights(const TinyJSON::TJValueObject& layer_object, unsigned layer_number);
  static std::vector<std::vector<WeightParam>> get_weights(const TinyJSON::TJValueObject& layer_object, unsigned layer_number);
  static std::vector<WeightParam> get_bias_weights(const TinyJSON::TJValueObject& layer_object, unsigned layer_number);
  static std::vector<Layer> create_layers(const NeuralNetworkOptions& options, const TinyJSON::TJValue& json);
  static const TinyJSON::TJValueObject* get_layer_object(const TinyJSON::TJValue& json, unsigned layer_number);
  static const TinyJSON::TJValueArray* get_layers_array(const TinyJSON::TJValue& json);
  static int get_number_of_layers(const TinyJSON::TJValue& json);

  static void add_basic(TinyJSON::TJValueObject& json);
  static void add_errors(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_layers(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_layer(const Layer& layer, TinyJSON::TJValueArray& layers);
  static TinyJSON::TJValueObject* add_neuron(const Neuron& neuron);
  static void add_weight_params(const std::vector<WeightParam>& weight_params, TinyJSON::TJValueObject& parent);
  static TinyJSON::TJValue* add_weight_param(const WeightParam& weight_param);
  static void add_options(const NeuralNetworkOptions& options, TinyJSON::TJValueObject& json);
  static void add_final_learning_rate(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);

  static void add_weights(const Layer& layer, TinyJSON::TJValueObject& layer_object);
  static void add_bias_weights(const Layer& layer, TinyJSON::TJValueObject& layer_object);
  static void add_residual_weights(const Layer& layer, TinyJSON::TJValueObject& layer_object);
};