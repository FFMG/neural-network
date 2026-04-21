#pragma once
#include <array>
#include <map>
#include <string>
#include <vector>

#include "branchedoutputlayer.h"
#include "elmanrnnlayer.h"
#include "errorcalculation.h"
#include "evaluationconfig.h"
#include "fflayer.h"
#include "ffoutputlayer.h"
#include "grurnnlayer.h"
#include "layer.h"
#include "layerdetails.h"
#include "layers.h"
#include "multioutputlayerdetails.h"
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
  virtual ~NeuralNetworkSerializer()
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  }

  static const std::vector<ErrorCalculation::type> all_error_types();
  static NeuralNetworkOptions get_and_build_options(const TinyJSON::TJValue& json);
  static std::vector<std::map<ErrorCalculation::type, double>> get_errors(const TinyJSON::TJValue& json);
  static std::vector<WeightParam> get_weight_params(const TinyJSON::TJValueObject& parent);
  static std::vector<Neuron> get_neurons(const TinyJSON::TJValue& json, unsigned layer_number);
  static std::vector<Neuron> get_neurons(const TinyJSON::TJValueObject& layer_object, unsigned layer_number);
  static Layers create_layers(const NeuralNetworkOptions& options, const TinyJSON::TJValue& json);
  static std::unique_ptr<Layer> create_fflayer(unsigned layer_index, const TinyJSON::TJValueObject& layer_object, int number_of_threads);
  static std::unique_ptr<Layer> create_ffoutputlayer(unsigned layer_index, const TinyJSON::TJValueObject& layer_object, int number_of_threads, const std::vector<OutputLayerDetails>& output_layer_details);
  static std::unique_ptr<Layer> create_elmanrnnlayer(unsigned layer_index, const TinyJSON::TJValueObject& layer_object, int number_of_threads);
  static std::unique_ptr<Layer> create_grurnnlayer(unsigned layer_index, const TinyJSON::TJValueObject& layer_object, int number_of_threads);
  static std::unique_ptr<Layer> create_multioutputlayer(unsigned layer_index, const TinyJSON::TJValueObject& layer_object, int number_of_threads, const std::vector<MultiOutputLayerDetails>& multi_output_layer_details);
  static const TinyJSON::TJValueObject* get_layer_object(const TinyJSON::TJValue& json, unsigned layer_number);
  static const TinyJSON::TJValueArray* get_layers_array(const TinyJSON::TJValue& json);
  static int get_number_of_layers(const TinyJSON::TJValue& json);
  static std::vector<LayerDetails> get_hidden_layers(const TinyJSON::TJValueObject& options_object);
  static std::vector<OutputLayerDetails> get_output_layer_details(const TinyJSON::TJValueObject& options_object);
  static std::vector<MultiOutputLayerDetails> get_multi_output_layer_details(const TinyJSON::TJValueObject& options_object);
  static EvaluationConfig get_error_evaluation_config(const TinyJSON::TJValueObject* parent);
  static layer_activation_helper get_activation_helper(const TinyJSON::TJValueObject& layer_object, unsigned num_inputs, unsigned num_outputs);
  static void add_error_evaluation_config(TinyJSON::TJValueObject* parent, const EvaluationConfig& config);

  static ResidualProjector* get_residual_projector(const TinyJSON::TJValueObject& layer_object);

  static void add_basic(TinyJSON::TJValueObject& json);
  static void add_errors(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_layers(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static void add_layer(const Layer* layer, TinyJSON::TJValueArray& layers);
  static void add_activation_helper(const layer_activation_helper& lah, TinyJSON::TJValueObject& json);
  static void add_fflayer(const FFLayer& layer, TinyJSON::TJValueArray& layers);
  static void add_ffoutputlayer(const FFOutputLayer& layer, TinyJSON::TJValueArray& layers);
  static void add_elmanrnnlayer(const ElmanRNNLayer& layer, TinyJSON::TJValueArray& layers);
  static void add_grurnnlayer(const GRURNNLayer& layer, TinyJSON::TJValueArray& layers);
  static void add_branchedoutputlayer(const BranchedOutputLayer& layer, TinyJSON::TJValueArray& layers);
  static TinyJSON::TJValueObject* add_neuron(const Neuron& neuron);
  static void add_weight_params(const std::vector<WeightParam>& weight_params, TinyJSON::TJValueObject& parent);
  static TinyJSON::TJValue* add_weight_param(const WeightParam& weight_param);
  static void load_weights(Layer& layer, const TinyJSON::TJValueObject& layer_object);
  static void add_options(const NeuralNetworkOptions& options, TinyJSON::TJValueObject& json);
  static void add_final_learning_rate(const NeuralNetwork& nn, TinyJSON::TJValueObject& json);
  static TinyJSON::TJValueObject* add_residual_projector(const ResidualProjector* residual_projector);
  static TinyJSON::TJValueArray* add_hidden_layers(const std::vector<LayerDetails>& hidden_layers);
  static TinyJSON::TJValueArray* add_output_layer_details(const std::vector<OutputLayerDetails>& output_layer_details);
};