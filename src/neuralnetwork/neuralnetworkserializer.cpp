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

  // get the options
  auto options = get_options(logger, *tj);

  // get the weights...
  std::vector<std::vector<Neuron>> array_of_neurons;
  for(auto layer_number = 0; ;++layer_number)
  {
    auto activation_method = options.hidden_activation_method();
    if (layer_number == 0)
    {
      activation_method = activation::method::linear;
    }
    if (layer_number == static_cast<int>(options.topology().size() - 1))
    {
      activation_method = options.output_activation_method();
    }

    auto neurons = get_neurons(logger, *tj, layer_number, activation_method);
    if(neurons.size() == 0)
    {
      break;
    }
    array_of_neurons.push_back(neurons);
  }

  auto residual_layers = get_residual_layers(logger, *tj);

  auto error = get_error(*tj);
  auto mean_absolute_percentage_error = get_mean_absolute_percentage_error(*tj);

  // create the layer and validate that the topology matches.
  auto layers = create_layers(logger, array_of_neurons, residual_layers);
  if(layers.size() == 0 )
  {
    logger.log_error("Found no valid layers to load!");
    delete tj;
    return nullptr;
  }

  // create the NN
  auto nn = new NeuralNetwork(layers, options);
  logger.log_info("Created Neural Network with Error: ", error, " and MAPE: ", (mean_absolute_percentage_error*100));

  // cleanup
  delete tj;
  return nn;
}

std::vector<Layer> NeuralNetworkSerializer::create_layers(Logger& logger, std::vector<std::vector<Neuron>> array_of_neurons, const std::vector<int>& residual_layers)
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
    const auto& residual_layer = residual_layers[i];
    const auto num_neurons_in_previous_layer = static_cast<unsigned>(array_of_neurons[i - 1].size());
    const auto& this_neurons = array_of_neurons[i];
    layers.emplace_back(Layer::create_hidden_layer(this_neurons, num_neurons_in_previous_layer, residual_layer, logger));
  }

  // finally, the output layer.
  const auto& residual_layer = residual_layers.back();
  auto output_neurons = array_of_neurons.back();
  const auto num_neurons_in_previous_layer = static_cast<unsigned>(array_of_neurons[array_of_neurons.size()-2].size());
  layers.emplace_back(Layer::create_output_layer(output_neurons, num_neurons_in_previous_layer, residual_layer, logger));
  return layers;
}

void NeuralNetworkSerializer::save(const NeuralNetwork& nn, const std::string& path)
{
  // create the object.
  auto tj = new TinyJSON::TJValueObject();
  add_basic(*tj);
  add_options(nn.options(), *tj);
  add_final_learning_rate(nn, *tj);
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

NeuralNetworkOptions NeuralNetworkSerializer::get_options(Logger& logger, const TinyJSON::TJValue& json)
{
  auto default_option = NeuralNetworkOptions::create({ 1,1 }).build();
  auto object = dynamic_cast<const TinyJSON::TJValueObject*>(&json);
  if (nullptr == object)
  {
    logger.log_error("The given json root is not an object!");
    return default_option;
  }
  auto options_object = dynamic_cast<const TinyJSON::TJValueObject*>(object->try_get_value("options"));
  if (nullptr == options_object)
  {
    logger.log_error("The given json does not contain a valid option section!");
    return default_option;
  }

  auto array = dynamic_cast<const TinyJSON::TJValueArray*>(options_object->try_get_value("topology"));
  if (nullptr == array)
  {
    logger.log_error("Could not find a 'topology' node!");
    return default_option;
  }

  std::vector<unsigned> topology;
  for (unsigned i = 0; i < array->get_number_of_items(); ++i)
  {
    auto number = dynamic_cast<const TinyJSON::TJValueNumberInt*>(array->at(i));
    if (nullptr == number)
    {
      logger.log_error("The 'topology' node does not have a valid number at position ", i, "!");
      return default_option;
    }
    topology.push_back(static_cast<unsigned>(number->get_number()));
  }
  
  auto hidden_activation_string = options_object->try_get_string("hidden-activation", false);
  auto output_activation_string = options_object->try_get_string("output-activation", false);
  auto hidden_activation = activation::string_to_method(hidden_activation_string);
  auto output_activation = activation::string_to_method(output_activation_string);
  
  auto learning_rate = options_object->get_float("learning-rate");
  auto number_of_epoch = static_cast<int>(options_object->get_number("number-of-epoch"));
  auto batch_size = static_cast<int>(options_object->get_number("batch-size"));
  auto data_is_unique = options_object->get_boolean("data-is-unique");
  auto number_of_threads = static_cast<int>(options_object->get_number("number-of-threads"));
  auto learning_rate_decay_rate = options_object->get_float("learning-rate-decay-rate");
  auto adaptive_learning_rate = options_object->get_boolean("adaptive-learning-rate");
  auto optimiser_type_string = options_object->try_get_string("optimiser-type");
  auto optimiser_type = string_to_optimiser_type(optimiser_type_string);

  auto learning_rate_restart_rate = options_object->get_float("learning-rate-restart-rate");
  auto learning_rate_restart_boost = options_object->get_float("learning-rate-restart-boost");
  auto residual_layer_jump = options_object->get_number("residual-layer-jump");

  return NeuralNetworkOptions::create(topology)
    .with_hidden_activation_method(hidden_activation)
    .with_output_activation_method(output_activation)
    .with_learning_rate(learning_rate)
    .with_number_of_epoch(number_of_epoch)
    .with_batch_size(batch_size)
    .with_data_is_unique(data_is_unique)
    .with_number_of_threads(number_of_threads)
    .with_learning_rate_decay_rate(learning_rate_decay_rate)
    .with_adaptive_learning_rates(adaptive_learning_rate)
    .with_optimiser_type(optimiser_type)
    .with_learning_rate_boost_rate(learning_rate_restart_rate, learning_rate_restart_boost)
    .with_residual_layer_jump(residual_layer_jump)
    .with_logger(logger)
    .build();
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

std::vector<int> NeuralNetworkSerializer::get_residual_layers(Logger& logger, const TinyJSON::TJValue& json)
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

  std::vector<int> residual_layers;
  unsigned total_number_of_residual_layers = array->get_number_of_items();
  for( unsigned i = 0; i < total_number_of_residual_layers; ++i)
  {
    auto layer_object = dynamic_cast<const TinyJSON::TJValueObject*>(array->at(i));
    if(nullptr == layer_object)
    {
      logger.log_warning("The 'layers' array did not contain valid layer objects!");
      return {};
    }
    auto residual_layer = static_cast<int>(layer_object->get_number("residual-layer", true, true));
    residual_layers.push_back(residual_layer);
  }
  return residual_layers;
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

  auto layer_object = dynamic_cast<const TinyJSON::TJValueObject*>(array->at(layer_number));
  if(nullptr == layer_object)
  {
    logger.log_error("Could not get layer object at position: ", layer_number);
    return {};
  }

  auto layer_array = dynamic_cast<const TinyJSON::TJValueArray*>(layer_object->try_get_value("neurons"));
  if(nullptr == layer_array)
  {
    logger.log_error("Layer object at position: ", layer_number, " does not contain a valid neuron node!");
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
      logger.log_error("Could not find neuron index!");
      return {};
    }
    auto optimiser_type_object = dynamic_cast<const TinyJSON::TJValueNumber*>(neuron_object->try_get_value("optimiser-type"));
    if (nullptr == optimiser_type_object)
    {
      logger.log_error("Could not find neuron optimiser type!");
      return {};
    }

    auto index = static_cast<unsigned>(index_object->get_number());
    auto optimiser_type = static_cast<OptimiserType>(optimiser_type_object->get_number());

    // then the weights
    // the output layer can have zero weights
    auto weight_params = get_weight_params(logger, *neuron_object);
    
    auto neuron = Neuron(
      index,
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

std::vector<WeightParam> NeuralNetworkSerializer::get_weight_params(Logger& logger, const TinyJSON::TJValueObject& neuron)
{
  // the array of weight
  auto weights_array = dynamic_cast<const TinyJSON::TJValueArray*>(neuron.try_get_value("weight-params"));
  if(nullptr == weights_array)
  {
    logger.log_error("Could not find a valid 'weights' node!");
    return {};
  }

  std::vector<WeightParam> weight_params;
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

    weight_params.emplace_back(WeightParam(
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


void NeuralNetworkSerializer::add_weight_params(const std::vector<WeightParam>& weight_params, TinyJSON::TJValueObject& neuron)
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

void NeuralNetworkSerializer::add_options(const NeuralNetworkOptions& options, TinyJSON::TJValueObject& json)
{
  auto options_object = new TinyJSON::TJValueObject();
  json.set("options", options_object);

  auto topology_list = new TinyJSON::TJValueArray();
  for (auto topology : options.topology())
  {
    topology_list->add_number(topology);
  }
  options_object->set("topology", topology_list);
  options_object->set_string("hidden-activation", activation::method_to_string(options.hidden_activation_method()).c_str());
  options_object->set_string("output-activation", activation::method_to_string(options.output_activation_method()).c_str());
  options_object->set_float("learning-rate", options.learning_rate());
  options_object->set_number("number-of-epoch", options.number_of_epoch());
  options_object->set_number("batch-size", options.batch_size());
  options_object->set_boolean("data-is-unique", options.data_is_unique());
  options_object->set_number("number-of-threads", options.number_of_threads());
  options_object->set_float("learning-rate-decay-rate", options.learning_rate_decay_rate());
  options_object->set_boolean("adaptive-learning-rate", options.adaptive_learning_rate());
  options_object->set_string("optimiser-type", optimiser_type_to_string(options.optimiser_type()).c_str());
  options_object->set_float("learning-rate-restart-rate", options.learning_rate_restart_rate());
  options_object->set_float("learning-rate-restart-boost", options.learning_rate_restart_boost());
  options_object->set_number("residual-layer-jump", options.residual_layer_jump());

  json.set("options", options_object);

  delete topology_list;
  delete options_object;
}

void NeuralNetworkSerializer::add_basic(TinyJSON::TJValueObject& json)
{
  auto now = std::chrono::system_clock::now();
  auto now_seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
  long long current_timestamp = now_seconds.time_since_epoch().count();

  json.set_number("created", current_timestamp);
}

TinyJSON::TJValueObject* NeuralNetworkSerializer::add_neuron(const Neuron& neuron)
{
  auto neuron_object = new TinyJSON::TJValueObject();
  neuron_object->set_number("index", neuron.get_index());
  neuron_object->set_number("optimiser-type", static_cast<unsigned>(neuron.get_optimiser_type()));
  add_weight_params(neuron.get_weight_params(), *neuron_object);
  return neuron_object;
}

void NeuralNetworkSerializer::add_layer(const Layer& layer, TinyJSON::TJValueArray& layers)
{
  auto layer_object = new TinyJSON::TJValueObject();
  auto layer_array = new TinyJSON::TJValueArray();
  for(auto neuron : layer.get_neurons())
  {
    auto* neuron_object = add_neuron(neuron);
    layer_array->add(neuron_object);
    delete neuron_object;
  }
  layer_object->set_number("residual-layer", layer.residual_layer_number());
  layer_object->set("neurons", layer_array);
  layers.add(layer_object);
  delete layer_array;
  delete layer_object;
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

void NeuralNetworkSerializer::add_final_learning_rate(const NeuralNetwork& nn, TinyJSON::TJValueObject& json)
{
  json.set_float("final-learning-rate", nn.get_learning_rate());
}