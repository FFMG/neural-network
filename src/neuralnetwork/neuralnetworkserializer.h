#pragma once
#include <array>
#include <string>
#include <vector>

#include "neuralnetwork.h"
#include "libraries/TinyJSON.h"

class NeuralNetworkSerializer
{
public:
  NeuralNetworkSerializer();
  NeuralNetworkSerializer( const NeuralNetworkSerializer& src) = delete;
  NeuralNetworkSerializer& operator=(const NeuralNetworkSerializer& src) = delete;
  virtual ~NeuralNetworkSerializer() = default;

  static NeuralNetwork* load(const std::string& path);
  static void save(const NeuralNetwork& nn, const std::string& path);

private:
  static double get_error(const TinyJSON::TJValue& json);
  static double get_learning_rate(const TinyJSON::TJValue& json);
  static std::vector<unsigned> get_topology(const TinyJSON::TJValue& json );
  static activation::method get_activation_method(const TinyJSON::TJValue& json );
  static Neuron::Layer* get_layer(
    const TinyJSON::TJValue& json, 
    unsigned layer_number,
    const activation::method& activation_method
  );
  static std::vector<std::array<double,2>> get_weights(const TinyJSON::TJValueObject& neuron);

  static void add_basic(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_topology(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_activation_method(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_error(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_layers(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_layer(const Neuron::Layer& layer, TinyJSON::TJValueArray& layers);
  static void add_neuron(const Neuron& neuron, TinyJSON::TJValueArray& layer);
  static void add_weights(const std::vector<std::array<double,2>>& weights, TinyJSON::TJValueObject& neuron);
};