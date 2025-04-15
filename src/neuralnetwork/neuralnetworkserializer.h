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
  static void add_topology(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_layers(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_layer(const Neuron::Layer& layer, TinyJSON::TJValueArray& layers);
  static void add_neuron(const Neuron& neuron, TinyJSON::TJValueArray& layer);
  static void add_weights(const std::vector<std::array<double,2>>& weights, TinyJSON::TJValueObject& neuron);
};