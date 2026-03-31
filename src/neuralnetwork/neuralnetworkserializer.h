#pragma once
#include <array>
#include <map>
#include <string>
#include <vector>

#include "errorcalculation.h"
#include "elmanrnnlayer.h"
#include "fflayer.h"
#include "grurnnlayer.h"
#include "layer.h"
#include "layerdetails.h"
#include "layers.h"
#include "neuralnetwork.h"
#include "neuron.h"
#include "outputlayerdetails.h"
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
  static std::vector<WeightParam> get_weight_params(const TinyJSON::TJValueObject& parent);
  static std::vector<Neuron> get_neurons(const TinyJSON::TJValue& json, unsigned layer_number);
  static std::vector<Neuron> get_neurons(const TinyJSON::TJValueObject& layer_object, unsigned layer_number);
  static Layers create_layers(const NeuralNetworkOptions& options, const TinyJSON::TJValue& json);
  static std::unique_ptr<Layer> create_fflayer(unsigned layer_index, const TinyJSON::TJValueObject& layer_object, int number_of_threads);
  static std::unique_ptr<Layer> create_elmanrnnlayer(unsigned layer_index, const TinyJSON::TJValueObject& layer_object, int number_of_threads);
  static std::unique_ptr<Layer> create_grurnnlayer(unsigned layer_index, const TinyJSON::TJValueObject& layer_object, int number_of_threads);
  static const TinyJSON::TJValueObject* get_layer_object(const TinyJSON::TJValue& json, unsigned layer_number);
  static const TinyJSON::TJValueArray* get_layers_array(const TinyJSON::TJValue& json);
  static int get_number_of_layers(const TinyJSON::TJValue& json);
  static std::vector<LayerDetails> get_hidden_layers(const TinyJSON::TJValueObject& options_object);
  static OutputLayerDetails get_output_layer_details(unsigned output_size, const TinyJSON::TJValueObject& options_object);
  static ErrorCalculation::EvaluationConfig get_error_evaluation_config(const TinyJSON::TJValueObject* parent);
  static void add_error_evaluation_config(TinyJSON::TJValueObject* parent, const ErrorCalculation::EvaluationConfig& config);

  static ResidualProjector* get_residual_projector(const TinyJSON::TJValueObject& layer_object);

  static void add_basic(TinyJSON::TJValueObject& json);
  static void add_errors(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_layers(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_fflayer(const FFLayer& layer, TinyJSON::TJValueArray& layers);
  static void add_elmanrnnlayer(const ElmanRNNLayer& layer, TinyJSON::TJValueArray& layers);
  static void add_grurnnlayer(const GRURNNLayer& layer, TinyJSON::TJValueArray& layers);
  static TinyJSON::TJValueObject* add_neuron(const Neuron& neuron);
  static void add_weight_params(const std::vector<WeightParam>& weight_params, TinyJSON::TJValueObject& parent);
  static TinyJSON::TJValue* add_weight_param(const WeightParam& weight_param);
  static void add_options(const NeuralNetworkOptions& options, TinyJSON::TJValueObject& json);
  static void add_final_learning_rate(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static TinyJSON::TJValueObject* add_residual_projector(const ResidualProjector* residual_projector);
  static TinyJSON::TJValueArray* add_hidden_layers(const std::vector<LayerDetails>& hidden_layers);
  static TinyJSON::TJValueObject* add_output_layer(const OutputLayerDetails& output_layer);
};