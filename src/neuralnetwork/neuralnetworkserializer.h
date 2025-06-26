#pragma once
#include <array>
#include <string>
#include <vector>

#include "layer.h"
#include "logger.h"
#include "neuralnetwork.h"
#include "neuron.h"
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

  static double get_error(const TinyJSON::TJValue& json);
  static double get_mean_absolute_percentage_error(const TinyJSON::TJValue& json);
  static std::vector<unsigned> get_topology(Logger& logger, const TinyJSON::TJValue& json);
  static activation::method get_hidden_activation_method(Logger& logger, const TinyJSON::TJValue& json );
  static activation::method get_output_activation_method(Logger& logger, const TinyJSON::TJValue& json);
  static std::vector<std::array<double,2>> get_weights(Logger& logger, const TinyJSON::TJValueObject& neuron);
  static std::vector<Neuron> get_neurons(Logger& logger, const TinyJSON::TJValue& json, unsigned layer_number,const activation::method& activation_method);
  static std::vector<Layer> create_layers(Logger& logger, std::vector<std::vector<Neuron>> array_of_neurons);

  static void add_basic(TinyJSON::TJValueObject& json);
  static void add_topology(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_activation_method(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_errors(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_layers(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_layer(const Layer& layer, TinyJSON::TJValueArray& layers);
  static void add_neuron(const Neuron& neuron, TinyJSON::TJValueArray& layer);
  static void add_weights(const std::vector<std::array<double,2>>& weights, TinyJSON::TJValueObject& neuron);
};