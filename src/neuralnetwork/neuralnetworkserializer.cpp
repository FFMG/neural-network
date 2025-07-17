#include <chrono>

#include "neuralnetworkserializer.h"
#include "optimiser.h"

NeuralNetworkSerializer::NeuralNetworkSerializer()
{
}

NeuralNetwork* NeuralNetworkSerializer::load(Logger& logger, const std::string& path)
{
  //  any errors will not throw, just return null
  TinyJSON::parse_options options_parse = {};
  options_parse.throw_exception = false;
  auto* tj = TinyJSON::TJ::parse_file(path.c_str(), options_parse);
  if(nullptr == tj)
  {
    logger.log_warning("Could not load Neural Network from file (", path, ").");
    return nullptr;
  }

  auto topology = get_topology(logger, *tj);
  auto hidden_activation_method = get_hidden_activation_method(logger, *tj);
  auto output_activation_method = get_output_activation_method(logger, *tj);

  // get the weights...
  std::vector<std::vector<Neuron>> array_of_neurons;
  for(auto layer_number = 0; ;++layer_number)
  {
    auto activation_method = hidden_activation_method;
    if (layer_number == 0)
    {
      activation_method = activation::method::linear;
    }
    if (layer_number == static_cast<int>(topology.size() -1))
    {
      activation_method = output_activation_method;
    }

    auto neurons = get_neurons(logger, *tj, layer_number, activation_method);
    if(neurons.size() == 0)
    {
      break;
    }
    array_of_neurons.push_back(neurons);
  }

  auto error = get_error(*tj);
  auto mean_absolute_percentage_error = get_mean_absolute_percentage_error(*tj);

  // create the layer and validate that the topology matches.
  auto layers = create_layers(logger, array_of_neurons);
  if(layers.size() == 0 )
  {
    logger.log_error("Found no valid layers to load!");
    delete tj;
    return nullptr;
  }

  // create the NN
  auto nn = new NeuralNetwork(layers, hidden_activation_method, output_activation_method, logger);
  logger.log_info("Created Neural Network with Error: ", error, " and MAPE: ", (mean_absolute_percentage_error*100));

  // cleanup
  delete tj;
  return nn;
}

std::vector<Layer> NeuralNetworkSerializer::create_layers(Logger& logger, std::vector<std::vector<Neuron>> array_of_neurons)
{
  std::vector<Layer> layers = {};
  auto number_of_layers = array_of_neurons.size();
  if(number_of_layers <= 2)
  {
    std::cerr << "The number of layers must be at least 2, (input+output)";
    return {};
  }

  layers.reserve(number_of_layers);

  // add the input layer
  auto input_neurons = array_of_neurons.front();
  layers.emplace_back(Layer::create_input_layer(input_neurons, logger));
  
  // create the hidden layers.
  for(size_t i = 1; i < number_of_layers -1; ++i)
  {
    const auto num_neurons_in_previous_layer = static_cast<unsigned>(array_of_neurons[i - 1].size());
    const auto& this_neurons = array_of_neurons[i];
    layers.emplace_back(Layer::create_hidden_layer(this_neurons, num_neurons_in_previous_layer, logger));
  }

  // finally, the output layer.
  auto output_neurons = array_of_neurons.back();
  const auto num_neurons_in_previous_layer = static_cast<unsigned>(array_of_neurons[array_of_neurons.size()-2].size());
  layers.emplace_back(Layer::create_output_layer(output_neurons, num_neurons_in_previous_layer, logger));
  return layers;
}

void NeuralNetworkSerializer::save(const NeuralNetwork& nn, const std::string& path)
{
  // create the object.
  auto tj = new TinyJSON::TJValueObject();
  add_basic(*tj);
  add_topology(nn, *tj);
  add_learning_rate(nn, *tj);
  add_activation_methods(nn, *tj);
  add_errors(nn, *tj);
  add_layers(nn, *tj);
  
  // save it.
  TinyJSON::TJ::write_file(path.c_str(), *tj);
  
  // cleanup
  delete tj;
}

double NeuralNetworkSerializer::get_mean_absolute_percentage_error(const TinyJSON::TJValue& json)
{
  auto object = dynamic_cast<const TinyJSON::TJValueObject*>(&json);
  if (nullptr == object)
  {
    return 0.0;
  }
  return object->get_float("mean-absolute-percentage-error", true, false);
}

double NeuralNetworkSerializer::get_error(const TinyJSON::TJValue& json)
{
  auto object = dynamic_cast<const TinyJSON::TJValueObject*>(&json);
  if (nullptr == object)
  {
    return 0.0;
  }
  return object->get_float("error", true, false);
}

double NeuralNetworkSerializer::get_learning_rate(const TinyJSON::TJValue& json)
{
  auto object = dynamic_cast<const TinyJSON::TJValueObject*>(&json);
  if (nullptr == object)
  {
    return 0.0;
  }
  return object->get_float("learning-rate", true, false);
}

std::vector<Neuron> NeuralNetworkSerializer::get_neurons(Logger& logger, const TinyJSON::TJValue& json, unsigned layer_number,const activation::method& activation_method)
{
  auto object = dynamic_cast<const TinyJSON::TJValueObject*>(&json);
  if(nullptr == object)
  {
    return {};
  }
  auto array = dynamic_cast<const TinyJSON::TJValueArray*>(object->try_get_value("layers"));
  if(nullptr == array)
  {
    return {};
  }
  if(layer_number >= unsigned(array->get_number_of_items()))
  {
    return {};
  }

  auto layer_array = dynamic_cast<const TinyJSON::TJValueArray*>(array->at(layer_number));
  if(nullptr == layer_array)
  {
    return {};
  }

  std::vector<Neuron> neurons;
  unsigned total_number_of_neurons = layer_array->get_number_of_items();
  for( unsigned i = 0; i < total_number_of_neurons; ++i)
  {
    auto neuron_object = dynamic_cast<const TinyJSON::TJValueObject*>(layer_array->at(i));
    if(nullptr == neuron_object)
    {
      return {};
    }
    auto index_object = dynamic_cast<const TinyJSON::TJValueNumber*>(neuron_object->try_get_value("index"));
    if(nullptr == index_object)
    {
      return {};
    }
    auto output_value_object = dynamic_cast<const TinyJSON::TJValueNumber*>(neuron_object->try_get_value("output"));
    if(nullptr == output_value_object)
    {
      return {};
    }
    auto optimiser_type_object = dynamic_cast<const TinyJSON::TJValueNumber*>(neuron_object->try_get_value("optimiser-type"));
    if (nullptr == optimiser_type_object)
    {
      return {};
    }

    auto index = static_cast<unsigned>(index_object->get_number());
    auto output_value = output_value_object->get_float();
    auto optimiser_type = static_cast<OptimiserType>(optimiser_type_object->get_number());

    // then the weights
    // the output layer can have zero weights
    auto weight_params = get_weight_params(logger, *neuron_object);
    
    auto neuron = Neuron(
      index,
      output_value,
      activation_method,
      weight_params,
      optimiser_type,
      i < total_number_of_neurons -1 ? Neuron::Type::Normal : Neuron::Type::Bias,
      logger
    );
    neurons.push_back(neuron);
  }
  return neurons;
}

std::vector<Neuron::WeightParam> NeuralNetworkSerializer::get_weight_params(Logger& logger, const TinyJSON::TJValueObject& neuron)
{
  // the array of weight
  auto weights_array = dynamic_cast<const TinyJSON::TJValueArray*>(neuron.try_get_value("weight-params"));
  if(nullptr == weights_array)
  {
    logger.log_error("Could not find a valid 'weights' node!");
    return {};
  }

  std::vector<Neuron::WeightParam> weight_params;
  weight_params.reserve(weights_array->get_number_of_items());
  for(unsigned i = 0; i < weights_array->get_number_of_items(); ++i)
  {
    auto weight_param_object = dynamic_cast<const TinyJSON::TJValueObject*>(weights_array->at(i));
    if( nullptr == weight_param_object)
    {
      logger.log_error("The 'weight-params' object was either empty or not a valid object!");
      return {};
    }
    auto value = weight_param_object->get_float("value");
    auto gradient = weight_param_object->get_float("gradient");
    auto velocity = weight_param_object->get_float("velocity");
    auto first_moment_estimate = weight_param_object->get_float("first-moment-estimate");
    auto second_moment_estimate = weight_param_object->get_float("second-moment-estimate");
    auto time_step = weight_param_object->get_number("time-step");
    auto weight_decay = weight_param_object->get_float("weight-decay");

    weight_params.emplace_back(Neuron::WeightParam(
      static_cast<double>(value),
      static_cast<double>(gradient),
      static_cast<double>(velocity),
      static_cast<double>(first_moment_estimate),
      static_cast<double>(second_moment_estimate),
      time_step,
      static_cast<double>(weight_decay),
      logger
    ));
  }
  return weight_params;
}

std::vector<unsigned> NeuralNetworkSerializer::get_topology(Logger& logger, const TinyJSON::TJValue& json)
{
  auto object = dynamic_cast<const TinyJSON::TJValueObject*>(&json);
  if(nullptr == object)
  {
    logger.log_error("Could not find a 'topology' node!");
    return {};
  }
  auto array = dynamic_cast<const TinyJSON::TJValueArray*>(object->try_get_value("topology"));
  if(nullptr == array)
  {
    logger.log_error("Could not find a 'topology' node!");
    return {};
  }

  std::vector<unsigned> topology;
  for( unsigned i = 0; i < array->get_number_of_items(); ++i)
  {
    auto number = dynamic_cast<const TinyJSON::TJValueNumberInt*>(array->at(i));
    if(nullptr == number)
    {
      logger.log_error("The 'topology' node does not have a valid number at position ", i, "!");
      return {};
    }
    topology.push_back(static_cast<unsigned>(number->get_number()));
  }

  if(topology.empty())
  {
    logger.log_error("The 'topology' node is empty!");
  }
  return topology;
}

activation::method NeuralNetworkSerializer::get_hidden_activation_method(Logger& logger, const TinyJSON::TJValue& json )
{
  auto object = dynamic_cast<const TinyJSON::TJValueObject*>(&json);
  if(nullptr == object)
  {
    logger.log_warning("Could not find a valid 'hidden-activation-method' node, defaulting to sigmoid.");
    return activation::method::sigmoid;
  }
  auto number = dynamic_cast<const TinyJSON::TJValueNumberInt*>(object->try_get_value("hidden-activation-method"));
  if(nullptr == number)
  {
    logger.log_warning("Could not find a valid 'hidden-activation-method' node, defaulting to sigmoid.");
    return activation::method::sigmoid;
  }
  return static_cast<activation::method>(number->get_number());
}

activation::method NeuralNetworkSerializer::get_output_activation_method(Logger& logger, const TinyJSON::TJValue& json)
{
  auto object = dynamic_cast<const TinyJSON::TJValueObject*>(&json);
  if (nullptr == object)
  {
    logger.log_warning("Could not find a valid 'output-activation-method' node, defaulting to sigmoid.");
    return activation::method::sigmoid;
  }
  auto number = dynamic_cast<const TinyJSON::TJValueNumberInt*>(object->try_get_value("output-activation-method"));
  if (nullptr == number)
  {
    logger.log_warning("Could not find a valid 'output-activation-method' node, defaulting to sigmoid.");
    return activation::method::sigmoid;
  }
  return static_cast<activation::method>(number->get_number());
}

void NeuralNetworkSerializer::add_weight_params(const std::vector<Neuron::WeightParam>& weight_params, TinyJSON::TJValueObject& neuron)
{
  auto weights_array = new TinyJSON::TJValueArray();
  for( auto weight_param : weight_params)
  {
    auto weight_param_object = new TinyJSON::TJValueObject();
    weight_param_object->set_float("value", weight_param.value());
    weight_param_object->set_float("gradient", weight_param.gradient());
    weight_param_object->set_float("velocity", weight_param.velocity());
    weight_param_object->set_float("first-moment-estimate", weight_param.first_moment_estimate());
    weight_param_object->set_float("second-moment-estimate", weight_param.second_moment_estimate());
    weight_param_object->set_number("time-step", weight_param.timestep());
    weight_param_object->set_float("weight-decay", weight_param.weight_decay());
    weights_array->add(weight_param_object);
    delete weight_param_object;
  }
  neuron.set("weight-params", weights_array);
  delete weights_array;
}

void NeuralNetworkSerializer::add_basic(TinyJSON::TJValueObject& json)
{
  auto now = std::chrono::system_clock::now();
  auto now_seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
  long long current_timestamp = now_seconds.time_since_epoch().count();

  json.set_number("created", current_timestamp);
}

void NeuralNetworkSerializer::add_neuron(const Neuron& neuron, TinyJSON::TJValueArray& layer)
{
  auto neuron_object = new TinyJSON::TJValueObject();
  neuron_object->set_number("index", neuron.get_index());
  neuron_object->set_number("optimiser-type", static_cast<unsigned>(neuron.get_optimiser_type()));
  add_weight_params(neuron.get_weight_params(), *neuron_object);
  layer.add(neuron_object);
  delete neuron_object;
}

void NeuralNetworkSerializer::add_layer(const Layer& layer, TinyJSON::TJValueArray& layers)
{
  auto layer_array = new TinyJSON::TJValueArray();
  for(auto neuron : layer.get_neurons())
  {
    add_neuron(neuron, *layer_array);
  }
  layers.add(layer_array);
  delete layer_array;
}

void NeuralNetworkSerializer::add_layers(const NeuralNetwork& nn, TinyJSON::TJValueObject& json)
{
  auto layers_array = new TinyJSON::TJValueArray();
  for(const auto& layer : nn.get_layers())
  {
    add_layer(layer, *layers_array);
  }
  json.set("layers", layers_array);
  delete layers_array;
}

void NeuralNetworkSerializer::add_errors(const NeuralNetwork& nn, TinyJSON::TJValueObject& json)
{
  auto metrics = nn.calculate_forecast_metrics({ NeuralNetworkOptions::ErrorCalculation::rmse, NeuralNetworkOptions::ErrorCalculation::mape });
  json.set_float("error", metrics[0].error());
  json.set_float("mean-absolute-percentage-error", metrics[0].error());
}

void NeuralNetworkSerializer::add_learning_rate(const NeuralNetwork& nn, TinyJSON::TJValueObject& json)
{
  json.set_float("learning-rate", nn.get_learning_rate());
}

void NeuralNetworkSerializer::add_activation_methods(const NeuralNetwork& nn, TinyJSON::TJValueObject& json)
{
  auto hidden_activation_method = new TinyJSON::TJValueNumberInt(static_cast<unsigned>(nn.get_hidden_activation_method()));
  json.set("hidden-activation-method", hidden_activation_method);
  delete hidden_activation_method;

  auto output_activation_method = new TinyJSON::TJValueNumberInt( static_cast<unsigned>(nn.get_output_activation_method()));
  json.set("output-activation-method", output_activation_method);
  delete output_activation_method;
}

void NeuralNetworkSerializer::add_topology(const NeuralNetwork& nn, TinyJSON::TJValueObject& json)
{
  auto topology_list = new TinyJSON::TJValueArray();
  for(auto topology : nn.get_topology())
  {
    topology_list->add_number(topology);
  }
  json.set("topology", topology_list);
  delete topology_list;
}