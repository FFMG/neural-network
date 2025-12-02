#include <chrono>

#include "logger.h"
#include "neuralnetworkserializer.h"
#include "optimiser.h"

NeuralNetworkSerializer::NeuralNetworkSerializer()
{
}

NeuralNetwork* NeuralNetworkSerializer::load(const std::string& path)
{
  //  any errors will not throw, just return null
  TinyJSON::parse_options options_parse = {};
  options_parse.throw_exception = false;
  auto* tj = TinyJSON::TJ::parse_file(path.c_str(), options_parse);
  if(nullptr == tj)
  {
    Logger::warning("Could not load Neural Network from file (", path, ").");
    return nullptr;
  }

  // get the options first so the logger is set
  auto options = get_and_build_options(*tj);

  auto errors = get_errors(*tj);

  // create the layer and validate that the topology matches.
  auto layers = create_layers(options, *tj);
  if(layers.size() == 0 )
  {
    Logger::error("Found no valid layers to load!");
    delete tj;
    return nullptr;
  }

  // create the NN
  auto nn = new NeuralNetwork(layers, options, errors);
  Logger::info("Created Neural Network.");

  // cleanup
  delete tj;
  return nn;
}

std::vector<Layer> NeuralNetworkSerializer::create_layers(const NeuralNetworkOptions& options, const TinyJSON::TJValue& json)
{
  auto number_of_layers = get_number_of_layers(json);
  if(number_of_layers <= 2)
  {
    Logger::error("The number of layers must be at least 2, (input+output)");
    return {};
  }

  std::vector<Layer> layers = {};
  layers.reserve(number_of_layers);
  
  // create the hidden layers.
  auto* layers_array = get_layers_array(json);
  if (nullptr == layers_array)
  {
    Logger::error("Could not locate the layers array.");
    return {};
  }

  unsigned number_input_neurons = 0;
  for(unsigned layer_index = 0; layer_index < static_cast<unsigned>(number_of_layers); ++layer_index)
  {
    auto neurons = get_neurons(json, layer_index);

    auto layer_object = static_cast<TinyJSON::TJValueObject*>(layers_array->at(layer_index));

    auto residual_layer_number = layer_object->get_number<int>("residual-layer-number");
    auto optimiser_type_string = layer_object->try_get_string("optimiser-type");
    if (optimiser_type_string == nullptr)
    {
      Logger::panic("Missing layer 'optimiser-type'.");
    }
    auto optimiser_type = string_to_optimiser_type(optimiser_type_string);

    auto activation_method_string = layer_object->try_get_string("activation-method");
    auto activation_method = activation::string_to_method(activation_method_string);

    auto layer_type_number = layer_object->get_number<int>("layer-type");
    auto layer_type = (Layer::LayerType)layer_type_number;

    auto layer = Layer(
      layer_index,
      neurons,
      number_input_neurons,
      residual_layer_number,
      layer_type,
      optimiser_type,
      activation_method,
      get_weights(*layer_object, layer_index),
      get_bias_weights(*layer_object, layer_index),
      get_residual_weights(*layer_object, layer_index)
    );
    layers.emplace_back(std::move(layer));

    number_input_neurons = static_cast<unsigned>(neurons.size());
  }
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

NeuralNetworkOptions NeuralNetworkSerializer::get_and_build_options(const TinyJSON::TJValue& json)
{
  auto default_option = NeuralNetworkOptions::create({ 1,1 }).build();
  auto object = dynamic_cast<const TinyJSON::TJValueObject*>(&json);
  if (nullptr == object)
  {
    Logger::error("The given json root is not an object!");
    return default_option;
  }
  auto options_object = dynamic_cast<const TinyJSON::TJValueObject*>(object->try_get_value("options"));
  if (nullptr == options_object)
  {
    Logger::error("The given json does not contain a valid option section!");
    return default_option;
  }

  auto array = dynamic_cast<const TinyJSON::TJValueArray*>(options_object->try_get_value("topology"));
  if (nullptr == array)
  {
    Logger::error("Could not find a 'topology' node!");
    return default_option;
  }

  std::vector<unsigned> topology;
  for (unsigned i = 0; i < array->get_number_of_items(); ++i)
  {
    auto number = dynamic_cast<const TinyJSON::TJValueNumberInt*>(array->at(i));
    if (nullptr == number)
    {
      Logger::error("The 'topology' node does not have a valid number at position ", i, "!");
      return default_option;
    }
    topology.push_back(static_cast<unsigned>(number->get_number()));
  }
  
  auto log_level_string = options_object->try_get_string("log-level", false);
  auto log_level = Logger::string_to_level(log_level_string);
  auto hidden_activation_string = options_object->try_get_string("hidden-activation", false);
  auto output_activation_string = options_object->try_get_string("output-activation", false);
  auto hidden_activation = activation::string_to_method(hidden_activation_string);
  auto output_activation = activation::string_to_method(output_activation_string);
  
  auto learning_rate = options_object->get_float<double>("learning-rate");
  auto learning_rate_warmup_start = options_object->get_float<double>("learning-rate-warmup-start");
  auto learning_rate_warmup_target = options_object->get_float<double>("learning-rate-warmup-target");

  auto number_of_epoch = static_cast<int>(options_object->get_number("number-of-epoch"));
  auto batch_size = static_cast<int>(options_object->get_number("batch-size"));
  auto data_is_unique = options_object->get_boolean("data-is-unique");
  auto number_of_threads = static_cast<int>(options_object->get_number("number-of-threads"));
  auto learning_rate_decay_rate = options_object->get_float<double>("learning-rate-decay-rate");
  auto adaptive_learning_rate = options_object->get_boolean("adaptive-learning-rate");
  auto optimiser_type_string = options_object->try_get_string("optimiser-type");
  auto optimiser_type = string_to_optimiser_type(optimiser_type_string);

  auto learning_rate_restart_rate = options_object->get_float<double>("learning-rate-restart-rate");
  auto learning_rate_restart_boost = options_object->get_float<double>("learning-rate-restart-boost");
  auto residual_layer_jump = static_cast<int>(options_object->get_number("residual-layer-jump"));
  auto clip_threshold = options_object->get_float<double>("clip-threshold");
  auto dropouts = options_object->get_floats<double>("dropout", false, false);
  auto shuffle_training_data = options_object->get_boolean("shuffle-training-data", false, false); 
  auto recurrent_layers = options_object->get_numbers<unsigned>("recurrent-layers", false, false);
  auto weight_decay = options_object->get_float<double>("weight-decay");

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
    .with_clip_threshold(clip_threshold)
    .with_learning_rate_warmup(learning_rate_warmup_start, learning_rate_warmup_target)
    .with_dropout(dropouts)
    .with_log_level(log_level)
    .with_shuffle_training_data(shuffle_training_data)
    .with_recurrent_layers(recurrent_layers)
    .with_weight_decay(weight_decay)
    .build();
}

std::map<ErrorCalculation::type, double> NeuralNetworkSerializer::get_errors(const TinyJSON::TJValue& json)
{
  std::map<ErrorCalculation::type, double> errors;
  auto tj_object = dynamic_cast<const TinyJSON::TJValueObject*>(&json);
  if (nullptr == tj_object)
  {
    Logger::info("Could not load any errors.");
    return errors;
  }
  auto tj_errors = dynamic_cast<const TinyJSON::TJValueObject*>(tj_object->try_get_value("errors"));
  if (nullptr == tj_errors)
  {
    Logger::info("Could not load any errors.");
    return errors;
  }

  errors[ErrorCalculation::type::huber_loss ] = tj_errors->get_float<double>("huber-loss", true, false);
  errors[ErrorCalculation::type::mae ] = tj_errors->get_float<double>("mae", true, false);
  errors[ErrorCalculation::type::mse ] = tj_errors->get_float<double>("mse", true, false);
  errors[ErrorCalculation::type::rmse ] = tj_errors->get_float<double>("rmse", true, false);
  errors[ErrorCalculation::type::nrmse ] = tj_errors->get_float<double>("nrmse", true, false);
  errors[ErrorCalculation::type::mape] = tj_errors->get_float<double>("mape", true, false);
  errors[ErrorCalculation::type::smape ] = tj_errors->get_float<double>("smape", true, false);
  errors[ErrorCalculation::type::wape ] = tj_errors->get_float<double>("wape", true, false);
  errors[ErrorCalculation::type::bce_loss ] = tj_errors->get_float<double>("bce-loss", true, false);
  errors[ErrorCalculation::type::directional_accuracy ] = tj_errors->get_float<double>("directional-accuracy", true, false);

  return errors;
}

const TinyJSON::TJValueArray* NeuralNetworkSerializer::get_layers_array(const TinyJSON::TJValue& json)
{
  auto object = dynamic_cast<const TinyJSON::TJValueObject*>(&json);
  if(nullptr == object)
  {
    Logger::error("Could not get layers array, the root is not an object!");
    return nullptr;
  }
  auto* array = dynamic_cast<const TinyJSON::TJValueArray*>(object->try_get_value("layers"));
  if(nullptr == array)
  {
    Logger::error("Could not get layers array!");
    return nullptr;
  }
  return array;
}

const TinyJSON::TJValueObject* NeuralNetworkSerializer::get_layer_object(const TinyJSON::TJValue& json, unsigned layer_number )
{
  auto* array = get_layers_array(json);
  if(nullptr == array)
  {
    return nullptr;
  }
  if(layer_number >= unsigned(array->get_number_of_items()))
  {
    return nullptr;
  }

  auto layer_object = dynamic_cast<const TinyJSON::TJValueObject*>(array->at(layer_number));
  if(nullptr == layer_object)
  {
    Logger::error("Could not get layer object at position: ", layer_number);
    return nullptr;
  }
  return layer_object;
}

std::vector<WeightParam> NeuralNetworkSerializer::get_bias_weights(const TinyJSON::TJValueObject& layer_object, unsigned layer_number)
{
  auto bias_object = dynamic_cast<const TinyJSON::TJValueObject*>(layer_object.try_get_value("bias-weights"));
  if (nullptr == bias_object)
  {
    // no residual layer...
    Logger::error("No bias weights for layer number: ", layer_number);
    return {};
  }
  auto size = bias_object->get_number<unsigned>("size", false, false);
  std::vector<WeightParam> all_weight_params;
  all_weight_params.reserve(size);
  return get_weight_params(*bias_object);
}

std::vector<std::vector<WeightParam>> NeuralNetworkSerializer::get_weights(const TinyJSON::TJValueObject& layer_object, unsigned layer_number)
{
  auto weights_array = dynamic_cast<const TinyJSON::TJValueArray*>(layer_object.try_get_value("weights"));
  if (nullptr == weights_array)
  {
    Logger::warning("Layer number: ", layer_number, " has no weights!");
    return {};
  }

  std::vector<std::vector<WeightParam>> all_weight_params;
  for(const auto& weight_array : *weights_array)
  {
    const auto* weight_array_object = static_cast<const TinyJSON::TJValueObject*>(&weight_array);
    auto weight_params = get_weight_params(*weight_array_object);
    all_weight_params.emplace_back(std::move(weight_params));
  }
  return all_weight_params;
}

std::vector<std::vector<WeightParam>> NeuralNetworkSerializer::get_residual_weights(const TinyJSON::TJValueObject& layer_object, unsigned layer_number)
{
  auto residual_projector_object = dynamic_cast<const TinyJSON::TJValueObject*>(layer_object.try_get_value("residual-projector"));
  if(nullptr == residual_projector_object)
  {
    // no residual layer...
    return {};
  }

  auto input_size = residual_projector_object->get_number("input-size", false, false);
  auto output_size = residual_projector_object->get_number("output-size", false, false);
  if(input_size <=0 || output_size <= 0)
  {
    // no residual layer... 
    return {};
  }

  auto residual_projector_array = dynamic_cast<const TinyJSON::TJValueArray*>(residual_projector_object->try_get_value("weight-params"));
  if(nullptr == residual_projector_array)
  {
    Logger::error("Layer object at position: ", layer_number, " does not contain a valid residual weight param node!");
    return {};
  }
  if(residual_projector_array->get_number_of_items() != output_size)
  {
    Logger::error("Layer object at position: ", layer_number, " residual weight param size does not match output count!");
    return {};
  }

  std::vector<std::vector<WeightParam>> all_weight_params;
  all_weight_params.reserve(output_size);
  for( unsigned i = 0; i < output_size; ++i)
  {
    auto all_weightparam_object = dynamic_cast<const TinyJSON::TJValueObject*>(residual_projector_array->at(i));
    if(nullptr == all_weightparam_object)
    {
      Logger::error("Layer object at position: ", layer_number, " unable to locate weight param at position ", i, "!");
      return {};
    }
    auto weight_params = get_weight_params(*all_weightparam_object);
    if(weight_params.size() != static_cast<size_t>(input_size))
    {
      Logger::error("Layer object at position: ", layer_number, " weight param at position ", i, " does not match the input size!");      
    }
    all_weight_params.emplace_back(std::move(weight_params));
  }
  return all_weight_params;
}

std::vector<Neuron> NeuralNetworkSerializer::get_neurons(const TinyJSON::TJValue& json, unsigned layer_number)
{
  const auto* layer_object = get_layer_object(json, layer_number);
  if(nullptr == layer_object)
  {
    return {};
  }

  auto layer_array = dynamic_cast<const TinyJSON::TJValueArray*>(layer_object->try_get_value("neurons"));
  if(nullptr == layer_array)
  {
    Logger::error("Layer object at position: ", layer_number, " does not contain a valid neuron node!");
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
      Logger::error("Could not find neuron index!");
      return {};
    }

    auto index = static_cast<unsigned>(index_object->get_number());

    auto neuron_type = static_cast<Neuron::Type>(neuron_object->get_number("neuron-type", true, true));
    auto dropout_rate = neuron_object->get_float<double>("dropout-rate", true, true);
    
    auto neuron = Neuron(
      index,
      neuron_type,
      dropout_rate
    );
    neurons.push_back(neuron);
  }
  return neurons;
}

int NeuralNetworkSerializer::get_number_of_layers(const TinyJSON::TJValue& json)
{
  auto* array = get_layers_array(json);
  if(nullptr == array)
  {
    return 0;
  }
  return array->get_number_of_items();
}

std::vector<WeightParam> NeuralNetworkSerializer::get_weight_params(const TinyJSON::TJValueObject& parent)
{
  // the array of weight
  auto weights_array = dynamic_cast<const TinyJSON::TJValueArray*>(parent.try_get_value("weight-params"));
  if(nullptr == weights_array)
  {
    Logger::error("Could not find a valid 'weights' node!");
    return {};
  }

  std::vector<WeightParam> weight_params;
  weight_params.reserve(weights_array->get_number_of_items());
  for(unsigned i = 0; i < weights_array->get_number_of_items(); ++i)
  {
    auto weight_param_object = dynamic_cast<const TinyJSON::TJValueObject*>(weights_array->at(i));
    if( nullptr == weight_param_object)
    {
      Logger::error("The 'weight-params' object was either empty or not a valid object!");
      return {};
    }
    auto value = weight_param_object->get_float("value");
    auto raw_gradient = weight_param_object->get_float("raw-gradient");
    auto unclipped_gradient = weight_param_object->get_float("unclipped-gradient");
    auto velocity = weight_param_object->get_float("velocity");
    auto first_moment_estimate = weight_param_object->get_float("first-moment-estimate");
    auto second_moment_estimate = weight_param_object->get_float("second-moment-estimate");
    auto time_step = weight_param_object->get_number("time-step");
    auto weight_decay = weight_param_object->get_float("weight-decay");

    weight_params.emplace_back(WeightParam(
      static_cast<double>(value),
      static_cast<double>(raw_gradient),
      static_cast<double>(unclipped_gradient),
      static_cast<double>(velocity),
      static_cast<double>(first_moment_estimate),
      static_cast<double>(second_moment_estimate),
      time_step,
      static_cast<double>(weight_decay)
    ));
  }
  return weight_params;
}

void NeuralNetworkSerializer::add_weight_params(
  const std::vector<WeightParam>& weight_params,
  TinyJSON::TJValueObject& parent)
{
  auto weights_array = new TinyJSON::TJValueArray();
  for( auto weight_param : weight_params)
  {
    auto weights_object = add_weight_param(weight_param);
    weights_array->add(weights_object);
    delete weights_object;
  }
  parent.set("weight-params", weights_array);
  delete weights_array;
}

TinyJSON::TJValue* NeuralNetworkSerializer::add_weight_param(const WeightParam& weight_param)
{
  auto weight_param_object = new TinyJSON::TJValueObject();
  weight_param_object->set_float("value", weight_param.get_value());
  weight_param_object->set_float("raw-gradient", weight_param.get_raw_gradient());
  weight_param_object->set_float("unclipped-gradient", weight_param.get_unclipped_gradient());
  weight_param_object->set_float("velocity", weight_param.get_velocity());
  weight_param_object->set_float("first-moment-estimate", weight_param.get_first_moment_estimate());
  weight_param_object->set_float("second-moment-estimate", weight_param.get_second_moment_estimate());
  weight_param_object->set_number("time-step", weight_param.get_timestep());
  weight_param_object->set_float("weight-decay", weight_param.get_weight_decay());
  return weight_param_object;
}

void NeuralNetworkSerializer::add_options(const NeuralNetworkOptions& options, TinyJSON::TJValueObject& json)
{
  auto options_object = new TinyJSON::TJValueObject();
  json.set("options", options_object);

  auto topology_list = new TinyJSON::TJValueArray();
  topology_list->add_numbers(options.topology());

  auto dropout_list = new TinyJSON::TJValueArray();
  dropout_list->add_floats(options.dropout());

  auto recurrent_list = new TinyJSON::TJValueArray();
  recurrent_list->add_numbers(options.recurrent_layers());

  options_object->set("topology", topology_list);
  options_object->set("recurrent-layers", recurrent_list);
  options_object->set("dropout", dropout_list);
  options_object->set_string("log-level", Logger::level_to_string(options.log_level()).c_str());
  options_object->set_string("hidden-activation", activation::method_to_string(options.hidden_activation_method()).c_str());
  options_object->set_string("output-activation", activation::method_to_string(options.output_activation_method()).c_str());
  options_object->set_float("learning-rate", options.learning_rate());
  options_object->set_float("learning-rate-warmup-start", options.learning_rate_warmup_start());
  options_object->set_float("learning-rate-warmup-target", options.learning_rate_warmup_target());
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
  options_object->set_float("clip-threshold", options.clip_threshold());
  options_object->set_boolean("shuffle-training-data", options.shuffle_training_data());
  options_object->set_float("weight-decay", options.weight_decay());

  json.set("options", options_object);

  delete dropout_list;
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

  neuron_object->set_number("neuron-type", static_cast<unsigned>(neuron.get_type()));
  if(neuron.is_dropout())
  {
    neuron_object->set_float("dropout-rate", neuron.get_dropout_rate());
  }
  else
  {
    neuron_object->set_float("dropout-rate", 0.0);
  }
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
  layer_object->set("neurons", layer_array);
  layer_object->set_number("residual-layer-number", layer.residual_layer_number());
  layer_object->set_string("optimiser-type", optimiser_type_to_string(layer.get_optimiser_type()).c_str());
  layer_object->set_string("activation-method", layer.get_activation().method_to_string().c_str());
  layer_object->set_number("layer-type", (int)layer.get_layer_type());

  add_weights(layer, *layer_object);
  add_bias_weights(layer, *layer_object);
  add_residual_weights(layer, *layer_object);

  layers.add(layer_object);
  delete layer_array;
  delete layer_object;
}

void NeuralNetworkSerializer::add_weights(const Layer& layer, TinyJSON::TJValueObject& layer_object)
{
  auto weights_object = new TinyJSON::TJValueObject();
  // input outputs
  weights_object->set_number("input-size", layer.get_number_input_neurons());
  weights_object->set_number("output-size", layer.get_number_output_neurons());

  auto input_weights_array = new TinyJSON::TJValueArray();
  const auto& weight_params = layer.get_weight_params();
  for (const auto& weight_param : weight_params)
  {
    auto input_weights_object = new TinyJSON::TJValueObject();
    add_weight_params(weight_param, *input_weights_object);
    input_weights_array->add(input_weights_object);
    delete input_weights_object;
  }
  layer_object.set("weights", input_weights_array);
  delete input_weights_array;
  delete weights_object;
}

void NeuralNetworkSerializer::add_bias_weights(const Layer& layer, TinyJSON::TJValueObject& layer_object)
{
  auto weights_object = new TinyJSON::TJValueObject();

  auto bias_weight_params = layer.get_bias_weight_params();

  // input outputs
  weights_object->set_number("size", bias_weight_params.size());

  add_weight_params(bias_weight_params, *weights_object);
  layer_object.set("bias-weights", weights_object);
  delete weights_object;
}

void NeuralNetworkSerializer::add_residual_weights(const Layer& layer, TinyJSON::TJValueObject& layer_object)
{
  auto residual_projector_object = new TinyJSON::TJValueObject();

  if(layer.residual_layer_number() != -1)
  {
    // input outputs
    residual_projector_object->set_number("input-size", layer.residual_input_size());
    residual_projector_object->set_number("output-size", layer.residual_output_size());

    // then all the weights
    auto* layer_weights_array = new TinyJSON::TJValueArray();
    const auto& layer_weights = layer.residual_weight_params();
    for( const auto& weights : layer_weights )
    {
      auto* weights_object = new TinyJSON::TJValueObject();
      add_weight_params(weights, *weights_object);
      layer_weights_array->add(weights_object);
      delete weights_object;
    }
    residual_projector_object->set("weight-params", layer_weights_array);
    delete layer_weights_array;
  }
  layer_object.set("residual-projector", residual_projector_object);
  delete residual_projector_object;
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
  auto metrics = nn.calculate_forecast_metrics({ 
    ErrorCalculation::type::huber_loss,
    ErrorCalculation::type::mae,
    ErrorCalculation::type::mse,
    ErrorCalculation::type::rmse,
    ErrorCalculation::type::nrmse,
    ErrorCalculation::type::mape,
    ErrorCalculation::type::smape,
    ErrorCalculation::type::wape,
    ErrorCalculation::type::directional_accuracy,
    ErrorCalculation::type::bce_loss
  });

  auto tj_errors = new TinyJSON::TJValueObject();
  tj_errors->set_float("huber-loss" , metrics[0].error());
  tj_errors->set_float("mae"        , metrics[1].error());
  tj_errors->set_float("mse"        , metrics[2].error());
  tj_errors->set_float("rmse"       , metrics[3].error());
  tj_errors->set_float("nrmse"      , metrics[4].error());
  tj_errors->set_float("mape"       , metrics[5].error());
  tj_errors->set_float("smape"      , metrics[6].error());
  tj_errors->set_float("wape"       , metrics[7].error());
  tj_errors->set_float("directional-accuracy", metrics[8].error());
  tj_errors->set_float("bce-loss"   , metrics[9].error());

  json.set("errors", tj_errors);
  delete tj_errors;
}

void NeuralNetworkSerializer::add_final_learning_rate(const NeuralNetwork& nn, TinyJSON::TJValueObject& json)
{
  json.set_float("final-learning-rate", nn.get_learning_rate());
}