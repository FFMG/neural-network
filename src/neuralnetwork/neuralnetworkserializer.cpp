#include <chrono>
#include <memory>

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
  options_parse.callback_function = [&](TinyJSON::parse_options::message_type message_type, const TJCHAR* exception_message) {
    if (message_type == TinyJSON::parse_options::fatal)
    {
      Logger::panic(exception_message);
    }
    else
    {
      Logger::warning(exception_message);
    }
    };

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

Layers NeuralNetworkSerializer::create_layers(
  const NeuralNetworkOptions& options, 
  const TinyJSON::TJValue& json
)
{
  std::vector<std::unique_ptr<Layer>> layers = {};
  auto number_of_layers = get_number_of_layers(json);
  if(number_of_layers <= 2)
  {
    Logger::error("The number of layers must be at least 2, (input+output)");
    return Layers(layers, 0);
  }

  layers.reserve(number_of_layers);
  
  // create the hidden layers.
  auto* layers_array = get_layers_array(json);
  if (nullptr == layers_array)
  {
    Logger::error("Could not locate the layers array.");
    return Layers(layers, 0);
  }

  unsigned number_input_neurons = 0;
  for(unsigned layer_index = 0; layer_index < static_cast<unsigned>(number_of_layers); ++layer_index)
  {
    const auto* layer_object = static_cast<const TinyJSON::TJValueObject*>(layers_array->at(layer_index));
    std::string type = layer_object->get_string("layer-name");
    if (type == "fflayer")
    {
      layers.emplace_back(
        std::move(create_fflayer(layer_index, *layer_object, options.number_of_threads()))
      );
      continue;
    }

    if (type == "elmanrnnlayer")
    {
      layers.emplace_back(
        std::move(create_elmanrnnlayer(layer_index, *layer_object, options.number_of_threads()))
      );
      continue;
    }

    if (type == "grurnnlayer")
    {
      layers.emplace_back(
        std::move(create_grurnnlayer(layer_index, *layer_object, options.number_of_threads()))
      );
      continue;
    }

    Logger::panic("Unknown Layer type:", type);
  }

  const auto* json_object = static_cast<const TinyJSON::TJValueObject*>(&json);
  auto weight_decay = json_object->get_float("layers-weight-decay");

  return Layers(layers, weight_decay);
}

std::unique_ptr<Layer> NeuralNetworkSerializer::create_elmanrnnlayer(
  unsigned layer_index,
  const TinyJSON::TJValueObject& layer_object,
  int number_of_threads
)
{
  // get the neurons
  auto neurons = get_neurons(layer_object, layer_index);

  auto residual_layer_number = layer_object.get_number<int>("residual-layer-number");
  auto optimiser_type_string = layer_object.try_get_string("optimiser-type");
  if (optimiser_type_string == nullptr)
  {
    Logger::panic("Missing layer 'optimiser-type'.");
  }
  auto optimiser_type = string_to_optimiser_type(optimiser_type_string);

  auto activation_method_string = layer_object.try_get_string("activation-method");
  auto activation_alpha = layer_object.get_float("activation-alpha", true, false);
  auto activation_method = activation(activation::string_to_method(activation_method_string), activation_alpha);

  auto layer_type_number = layer_object.get_number<int>("layer-type");
  auto layer_type = (Layer::LayerType)layer_type_number;

  auto number_input_neurons = layer_object.get_number<unsigned>("number-input-neurons");
  auto number_output_neurons = layer_object.get_number<unsigned>("number-output-neurons");
  auto w_values = layer_object.get_floats<double>("w-values");
  auto w_grads = layer_object.get_floats<double>("w-grads");
  auto w_velocities = layer_object.get_floats<double>("w-velocities");
  auto w_m1 = layer_object.get_floats<double>("w-m1");
  auto w_m2 = layer_object.get_floats<double>("w-m2");
  auto w_timesteps = layer_object.get_numbers<long long>("w-timesteps");
  auto w_decays = layer_object.get_floats<double>("w-decays");

  auto b_values = layer_object.get_floats<double>("b-values");
  auto b_grads = layer_object.get_floats<double>("b-grads");
  auto b_velocities = layer_object.get_floats<double>("b-velocities");
  auto b_m1 = layer_object.get_floats<double>("b-m1");
  auto b_m2 = layer_object.get_floats<double>("b-m2");
  auto b_timesteps = layer_object.get_numbers<long long>("b-timesteps");
  auto b_decays = layer_object.get_floats<double>("b-decays");

  auto rw_values = layer_object.get_floats<double>("rw-values");
  auto rw_grads = layer_object.get_floats<double>("rw-grads");
  auto rw_velocities = layer_object.get_floats<double>("rw-velocities");
  auto rw_m1 = layer_object.get_floats<double>("rw-m1");
  auto rw_m2 = layer_object.get_floats<double>("rw-m2");
  auto rw_timesteps = layer_object.get_numbers<long long>("rw-timesteps");
  auto rw_decays = layer_object.get_floats<double>("rw-decays");

  auto residual_projector = get_residual_projector(layer_object);

  auto layer = std::make_unique<ElmanRNNLayer>(
    layer_index,
    layer_type,
    activation_method,
    optimiser_type,
    residual_layer_number,
    number_input_neurons,
    number_output_neurons,
    neurons,
    w_values,
    w_grads,
    w_velocities,
    w_m1,
    w_m2,
    w_timesteps,
    w_decays,
    b_values,
    b_grads,
    b_velocities,
    b_m1,
    b_m2,
    b_timesteps,
    b_decays,
    rw_values,
    rw_grads,
    rw_velocities,
    rw_m1,
    rw_m2,
    rw_timesteps,
    rw_decays,
    residual_projector,
    number_of_threads
  );

  return layer;
}

std::unique_ptr<Layer> NeuralNetworkSerializer::create_grurnnlayer(
  unsigned layer_index,
  const TinyJSON::TJValueObject& layer_object,
  int number_of_threads
)
{
  // get the neurons
  auto neurons = get_neurons(layer_object, layer_index);

  auto residual_layer_number = layer_object.get_number<int>("residual-layer-number");
  auto optimiser_type_string = layer_object.try_get_string("optimiser-type");
  if (optimiser_type_string == nullptr)
  {
    Logger::panic("Missing layer 'optimiser-type'.");
  }
  auto optimiser_type = string_to_optimiser_type(optimiser_type_string);

  auto activation_method_string = layer_object.try_get_string("activation-method");
  auto activation_alpha = layer_object.get_float("activation-alpha", true, false);
  auto activation_method = activation(activation::string_to_method(activation_method_string), activation_alpha);

  auto layer_type_number = layer_object.get_number<int>("layer-type");
  auto layer_type = (Layer::LayerType)layer_type_number;

  auto number_input_neurons = layer_object.get_number<unsigned>("number-input-neurons");
  auto number_output_neurons = layer_object.get_number<unsigned>("number-output-neurons");
  auto w_values = layer_object.get_floats<double>("w-values");
  auto w_grads = layer_object.get_floats<double>("w-grads");
  auto w_velocities = layer_object.get_floats<double>("w-velocities");
  auto w_m1 = layer_object.get_floats<double>("w-m1");
  auto w_m2 = layer_object.get_floats<double>("w-m2");
  auto w_timesteps = layer_object.get_numbers<long long>("w-timesteps");
  auto w_decays = layer_object.get_floats<double>("w-decays");

  auto b_values = layer_object.get_floats<double>("b-values");
  auto b_grads = layer_object.get_floats<double>("b-grads");
  auto b_velocities = layer_object.get_floats<double>("b-velocities");
  auto b_m1 = layer_object.get_floats<double>("b-m1");
  auto b_m2 = layer_object.get_floats<double>("b-m2");
  auto b_timesteps = layer_object.get_numbers<long long>("b-timesteps");
  auto b_decays = layer_object.get_floats<double>("b-decays");

  auto rw_values = layer_object.get_floats<double>("rw-values");
  auto rw_grads = layer_object.get_floats<double>("rw-grads");
  auto rw_velocities = layer_object.get_floats<double>("rw-velocities");
  auto rw_m1 = layer_object.get_floats<double>("rw-m1");
  auto rw_m2 = layer_object.get_floats<double>("rw-m2");
  auto rw_timesteps = layer_object.get_numbers<long long>("rw-timesteps");
  auto rw_decays = layer_object.get_floats<double>("rw-decays");

  auto z_w_values = layer_object.get_floats<double>("z-w-values");
  auto z_w_grads = layer_object.get_floats<double>("z-w-grads");
  auto z_w_velocities = layer_object.get_floats<double>("z-w-velocities");
  auto z_w_m1 = layer_object.get_floats<double>("z-w-m1");
  auto z_w_m2 = layer_object.get_floats<double>("z-w-m2");
  auto z_w_timesteps = layer_object.get_numbers<long long>("z-w-timesteps");
  auto z_w_decays = layer_object.get_floats<double>("z-w-decays");

  auto z_rw_values = layer_object.get_floats<double>("z-rw-values");
  auto z_rw_grads = layer_object.get_floats<double>("z-rw-grads");
  auto z_rw_velocities = layer_object.get_floats<double>("z-rw-velocities");
  auto z_rw_m1 = layer_object.get_floats<double>("z-rw-m1");
  auto z_rw_m2 = layer_object.get_floats<double>("z-rw-m2");
  auto z_rw_timesteps = layer_object.get_numbers<long long>("z-rw-timesteps");
  auto z_rw_decays = layer_object.get_floats<double>("z-rw-decays");

  auto z_b_values = layer_object.get_floats<double>("z-b-values");
  auto z_b_grads = layer_object.get_floats<double>("z-b-grads");
  auto z_b_velocities = layer_object.get_floats<double>("z-b-velocities");
  auto z_b_m1 = layer_object.get_floats<double>("z-b-m1");
  auto z_b_m2 = layer_object.get_floats<double>("z-b-m2");
  auto z_b_timesteps = layer_object.get_numbers<long long>("z-b-timesteps");
  auto z_b_decays = layer_object.get_floats<double>("z-b-decays");

  auto r_w_values = layer_object.get_floats<double>("r-w-values");
  auto r_w_grads = layer_object.get_floats<double>("r-w-grads");
  auto r_w_velocities = layer_object.get_floats<double>("r-w-velocities");
  auto r_w_m1 = layer_object.get_floats<double>("r-w-m1");
  auto r_w_m2 = layer_object.get_floats<double>("r-w-m2");
  auto r_w_timesteps = layer_object.get_numbers<long long>("r-w-timesteps");
  auto r_w_decays = layer_object.get_floats<double>("r-w-decays");

  auto r_rw_values = layer_object.get_floats<double>("r-rw-values");
  auto r_rw_grads = layer_object.get_floats<double>("r-rw-grads");
  auto r_rw_velocities = layer_object.get_floats<double>("r-rw-velocities");
  auto r_rw_m1 = layer_object.get_floats<double>("r-rw-m1");
  auto r_rw_m2 = layer_object.get_floats<double>("r-rw-m2");
  auto r_rw_timesteps = layer_object.get_numbers<long long>("r-rw-timesteps");
  auto r_rw_decays = layer_object.get_floats<double>("r-rw-decays");

  auto r_b_values = layer_object.get_floats<double>("r-b-values");
  auto r_b_grads = layer_object.get_floats<double>("r-b-grads");
  auto r_b_velocities = layer_object.get_floats<double>("r-b-velocities");
  auto r_b_m1 = layer_object.get_floats<double>("r-b-m1");
  auto r_b_m2 = layer_object.get_floats<double>("r-b-m2");
  auto r_b_timesteps = layer_object.get_numbers<long long>("r-b-timesteps");
  auto r_b_decays = layer_object.get_floats<double>("r-b-decays");

  auto residual_projector = get_residual_projector(layer_object);

  auto layer = std::make_unique<GRURNNLayer>(
    layer_index,
    layer_type,
    activation_method,
    optimiser_type,
    residual_layer_number,
    number_input_neurons,
    number_output_neurons,
    neurons,
    w_values,
    w_grads,
    w_velocities,
    w_m1,
    w_m2,
    w_timesteps,
    w_decays,
    b_values,
    b_grads,
    b_velocities,
    b_m1,
    b_m2,
    b_timesteps,
    b_decays,
    rw_values,
    rw_grads,
    rw_velocities,
    rw_m1,
    rw_m2,
    rw_timesteps,
    rw_decays,
    // Update Gate (z)
    z_w_values,
    z_w_grads,
    z_w_velocities,
    z_w_m1,
    z_w_m2,
    z_w_timesteps,
    z_w_decays,
    z_rw_values,
    z_rw_grads,
    z_rw_velocities,
    z_rw_m1,
    z_rw_m2,
    z_rw_timesteps,
    z_rw_decays,
    z_b_values,
    z_b_grads,
    z_b_velocities,
    z_b_m1,
    z_b_m2,
    z_b_timesteps,
    z_b_decays,
    // Reset Gate (r)
    r_w_values,
    r_w_grads,
    r_w_velocities,
    r_w_m1,
    r_w_m2,
    r_w_timesteps,
    r_w_decays,
    r_rw_values,
    r_rw_grads,
    r_rw_velocities,
    r_rw_m1,
    r_rw_m2,
    r_rw_timesteps,
    r_rw_decays,
    r_b_values,
    r_b_grads,
    r_b_velocities,
    r_b_m1,
    r_b_m2,
    r_b_timesteps,
    r_b_decays,
    residual_projector,
    number_of_threads
  );

  return layer;
}

std::unique_ptr<Layer> NeuralNetworkSerializer::create_fflayer(
  unsigned layer_index,
  const TinyJSON::TJValueObject& layer_object,
  int number_of_threads
)
{
  // get the neurons
  auto neurons = get_neurons(layer_object, layer_index);

  auto residual_layer_number = layer_object.get_number<int>("residual-layer-number");
  auto optimiser_type_string = layer_object.try_get_string("optimiser-type");
  if (optimiser_type_string == nullptr)
  {
    Logger::panic("Missing layer 'optimiser-type'.");
  }
  auto optimiser_type = string_to_optimiser_type(optimiser_type_string);

  auto activation_method_string = layer_object.try_get_string("activation-method");
  auto activation_alpha = layer_object.get_float("activation-alpha", true, false);
  auto activation_method = activation(activation::string_to_method(activation_method_string), activation_alpha);

  auto layer_type_number = layer_object.get_number<int>("layer-type");
  auto layer_type = (Layer::LayerType)layer_type_number;

  auto number_input_neurons = layer_object.get_number<unsigned>("number-input-neurons");
  auto number_output_neurons = layer_object.get_number<unsigned>("number-output-neurons");
  auto w_values = layer_object.get_floats<double>("w-values");
  auto w_grads = layer_object.get_floats<double>("w-grads");
  auto w_velocities = layer_object.get_floats<double>("w-velocities");
  auto w_m1 = layer_object.get_floats<double>("w-m1");
  auto w_m2 = layer_object.get_floats<double>("w-m2");
  auto w_timesteps = layer_object.get_numbers<long long>("w-timesteps");
  auto w_decays = layer_object.get_floats<double>("w-decays");
  auto b_values = layer_object.get_floats<double>("b-values");
  auto b_grads = layer_object.get_floats<double>("b-grads");
  auto b_velocities = layer_object.get_floats<double>("b-velocities");
  auto b_m1 = layer_object.get_floats<double>("b-m1");
  auto b_m2 = layer_object.get_floats<double>("b-m2");
  auto b_timesteps = layer_object.get_numbers<long long>("b-timesteps");
  auto b_decays = layer_object.get_floats<double>("b-decays");

  auto residual_projector = get_residual_projector(layer_object);

  auto layer = std::make_unique<FFLayer>(
    layer_index,
    layer_type,
    activation_method,
    optimiser_type,
    residual_layer_number,
    number_input_neurons,
    number_output_neurons,
    neurons,
    w_values,
    w_grads,
    w_velocities,
    w_m1,
    w_m2,
    w_timesteps,
    w_decays,
    b_values,
    b_grads,
    b_velocities,
    b_m1,
    b_m2,
    b_timesteps,
    b_decays,
    residual_projector,
    number_of_threads
  );

  return layer;
}

std::vector<LayerDetails> NeuralNetworkSerializer::get_hidden_layers(const TinyJSON::TJValueObject& options_object)
{
  const auto* hidden_layers_array = static_cast<const TinyJSON::TJValueArray*>(options_object.try_get_value("hidden-layers"));
  if (nullptr == hidden_layers_array)
  {
    Logger::panic("Could not find hidden layers option!");
  }
  std::vector<LayerDetails> hidden_layer;
  hidden_layer.reserve(hidden_layers_array->get_number_of_items());
  for (const auto& hlo : *hidden_layers_array)
  {
    const auto* phlo = static_cast<const TinyJSON::TJValueObject*>(&hlo);
    if (nullptr == phlo)
    {
      Logger::panic("Invalid hidden layer option is not an object!");
    }
    hidden_layer.emplace_back(LayerDetails(
      LayerDetails::type_from_string(phlo->try_get_string("type")), 
      phlo->get_number<unsigned>("size")
    ));
  }
  return hidden_layer;
}

ResidualProjector* NeuralNetworkSerializer::get_residual_projector(const TinyJSON::TJValueObject& layer_object)
{
  const auto* residual_projector_object = static_cast<const TinyJSON::TJValueObject*>(layer_object.try_get_value("residual-projector"));
  if (nullptr == residual_projector_object)
  {
    return nullptr;
  }

  auto input_size = residual_projector_object->get_number<unsigned>("input-size", false, false);
  auto output_size = residual_projector_object->get_number<unsigned>("output-size", false, false);
  auto w_values = residual_projector_object->get_floats<double>("w-values");
  auto w_grads = residual_projector_object->get_floats<double>("w-grads");
  auto w_velocities = residual_projector_object->get_floats<double>("w-velocities");
  auto w_m1 = residual_projector_object->get_floats<double>("w-m1");
  auto w_m2 = residual_projector_object->get_floats<double>("w-m2");
  auto w_timesteps = residual_projector_object->get_numbers<long long>("w-timesteps");
  auto w_decays = residual_projector_object->get_floats<double>("w-decays");

  return new ResidualProjector(
    input_size,
    output_size,
    w_values,
    w_grads,
    w_velocities,
    w_m1,
    w_m2,
    w_timesteps,
    w_decays
  );
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
  auto hidden_activation_alpha = options_object->get_float<double>("hidden-activation-alpha", true, false);
  auto output_activation_alpha = options_object->get_float<double>("output-activation-alpha", true, false);
  
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
  auto hidden_layers = get_hidden_layers(*options_object);
  auto weight_decay = options_object->get_float<double>("weight-decay");
  
  auto enable_bptt = options_object->get_boolean("enable-bptt", false, false);
  int bptt_max_ticks = options_object->get_number<int>("bptt-max-ticks");
  auto shuffle_bptt_batches = options_object->get_boolean("shuffle-bptt-batches", false, false);

  auto output_error_calculation_type_string = options_object->try_get_string("output-error-calculation-type");

  auto output_error_calculation_type = ErrorCalculation::string_to_type(output_error_calculation_type_string == nullptr ? "mse" : output_error_calculation_type_string);

  return NeuralNetworkOptions::create(topology)
    .with_hidden_activation_method(hidden_activation)
    .with_output_activation_method(output_activation)
    .with_hidden_activation_alpha(hidden_activation_alpha)
    .with_output_activation_alpha(output_activation_alpha)
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
    .with_hidden_layers(hidden_layers)
    .with_weight_decay(weight_decay)
    .with_bptt_max_ticks(bptt_max_ticks)
    .with_shuffle_bptt_batches(shuffle_bptt_batches)
    .with_output_error_calculation_type(output_error_calculation_type)
    .with_enable_bptt(enable_bptt)
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
  errors[ErrorCalculation::type::directional_accuracy] = tj_errors->get_float<double>("directional-accuracy", true, false);
  errors[ErrorCalculation::type::bce_loss ] = tj_errors->get_float<double>("bce-loss", true, false);
  errors[ErrorCalculation::type::cross_entropy] = tj_errors->get_float<double>("cross-entropy", true, false);

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

std::vector<Neuron> NeuralNetworkSerializer::get_neurons(const TinyJSON::TJValue& json, unsigned layer_number)
{
  const auto* layer_object = get_layer_object(json, layer_number);
  if (nullptr == layer_object)
  {
    return {};
  }
  return get_neurons(*layer_object, layer_number);
}

std::vector<Neuron> NeuralNetworkSerializer::get_neurons(const TinyJSON::TJValueObject& layer_object, unsigned layer_number)
{
  auto layer_array = dynamic_cast<const TinyJSON::TJValueArray*>(layer_object.try_get_value("neurons"));
  if (nullptr == layer_array)
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
    auto velocity = weight_param_object->get_float("velocity");
    auto first_moment_estimate = weight_param_object->get_float("first-moment-estimate");
    auto second_moment_estimate = weight_param_object->get_float("second-moment-estimate");
    auto time_step = weight_param_object->get_number("time-step");
    auto weight_decay = weight_param_object->get_float("weight-decay");

    weight_params.emplace_back(WeightParam(
      static_cast<double>(value),
      static_cast<double>(raw_gradient),
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
  
  auto topology_list = new TinyJSON::TJValueArray();
  topology_list->add_numbers(options.topology());

  auto dropout_list = new TinyJSON::TJValueArray();
  dropout_list->add_floats(options.dropout());

  auto hidden_layer_list = add_hidden_layers(options.hidden_layers());

  options_object->set("topology", topology_list);
  options_object->set("hidden-layers", hidden_layer_list);
  options_object->set("dropout", dropout_list);
  options_object->set_string("log-level", Logger::level_to_string(options.log_level()).c_str());
  options_object->set_string("hidden-activation", activation::method_to_string(options.hidden_activation_method()).c_str());
  options_object->set_string("output-activation", activation::method_to_string(options.output_activation_method()).c_str());
  options_object->set_float("hidden-activation-alpha", options.hidden_activation_alpha());
  options_object->set_float("output-activation-alpha", options.output_activation_alpha());
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
  options_object->set_number("bptt-max-ticks", options.bptt_max_ticks());
  options_object->set_string("output-error-calculation-type", ErrorCalculation::type_to_string(options.output_error_calculation_type()).c_str());
  options_object->set_boolean("enable-bptt", options.enable_bptt());
  options_object->set_boolean("shuffle-bptt-batches", options.shuffle_bptt_batches());

  json.set("options", options_object);

  delete hidden_layer_list;
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

void NeuralNetworkSerializer::add_elmanrnnlayer(const ElmanRNNLayer& layer, TinyJSON::TJValueArray& layers)
{
  auto layer_object = new TinyJSON::TJValueObject();
  auto layer_array = new TinyJSON::TJValueArray();
  for (const auto& neuron : layer.get_neurons())
  {
    auto* neuron_object = add_neuron(neuron);
    layer_array->add(neuron_object);
    delete neuron_object;
  }
  layer_object->set_string("layer-name", "elmanrnnlayer");
  layer_object->set("neurons", layer_array);
  layer_object->set_number("residual-layer-number", layer.get_residual_layer_number());
  layer_object->set_string("optimiser-type", optimiser_type_to_string(layer.get_optimiser_type()).c_str());
  layer_object->set_string("activation-method", layer.get_activation().method_to_string().c_str());
  layer_object->set_float("activation-alpha", layer.get_activation().get_alpha());
  layer_object->set_number("layer-type", (int)layer.get_layer_type());

  layer_object->set_number("number-input-neurons", layer.get_number_input_neurons());
  layer_object->set_number("number-output-neurons", layer.get_number_output_neurons());
  layer_object->set_floats("w-values", layer.get_w_values());
  layer_object->set_floats("w-grads", layer.get_w_grads());
  layer_object->set_floats("w-velocities", layer.get_w_velocities());
  layer_object->set_floats("w-m1", layer.get_w_m1());
  layer_object->set_floats("w-m2", layer.get_w_m2());
  layer_object->set_numbers("w-timesteps", layer.get_w_timesteps());
  layer_object->set_floats("w-decays", layer.get_w_decays());

  layer_object->set_floats("b-values", layer.get_b_values());
  layer_object->set_floats("b-grads", layer.get_b_grads());
  layer_object->set_floats("b-velocities", layer.get_b_velocities());
  layer_object->set_floats("b-m1", layer.get_b_m1());
  layer_object->set_floats("b-m2", layer.get_b_m2());
  layer_object->set_numbers("b-timesteps", layer.get_b_timesteps());
  layer_object->set_floats("b-decays", layer.get_b_decays());

  layer_object->set_floats("rw-values", layer.get_rw_values());
  layer_object->set_floats("rw-grads", layer.get_rw_grads());
  layer_object->set_floats("rw-velocities", layer.get_rw_velocities());
  layer_object->set_floats("rw-m1", layer.get_rw_m1());
  layer_object->set_floats("rw-m2", layer.get_rw_m2());
  layer_object->set_numbers("rw-timesteps", layer.get_rw_timesteps());
  layer_object->set_floats("rw-decays", layer.get_rw_decays());

  auto residual_projector = add_residual_projector(layer.get_residual_projector());
  if (residual_projector != nullptr)
  {
    layer_object->set("residual-projector", residual_projector);
    delete residual_projector;
  }

  layers.add(layer_object);
  delete layer_array;
  delete layer_object;
}

void NeuralNetworkSerializer::add_grurnnlayer(const GRURNNLayer& layer, TinyJSON::TJValueArray& layers)
{
  auto layer_object = new TinyJSON::TJValueObject();
  auto layer_array = new TinyJSON::TJValueArray();
  for (const auto& neuron : layer.get_neurons())
  {
    auto* neuron_object = add_neuron(neuron);
    layer_array->add(neuron_object);
    delete neuron_object;
  }
  layer_object->set_string("layer-name", "grurnnlayer");
  layer_object->set("neurons", layer_array);
  layer_object->set_number("residual-layer-number", layer.get_residual_layer_number());
  layer_object->set_string("optimiser-type", optimiser_type_to_string(layer.get_optimiser_type()).c_str());
  layer_object->set_string("activation-method", layer.get_activation().method_to_string().c_str());
  layer_object->set_float("activation-alpha", layer.get_activation().get_alpha());
  layer_object->set_number("layer-type", (int)layer.get_layer_type());

  layer_object->set_number("number-input-neurons", layer.get_number_input_neurons());
  layer_object->set_number("number-output-neurons", layer.get_number_output_neurons());
  layer_object->set_floats("w-values", layer.get_w_values());
  layer_object->set_floats("w-grads", layer.get_w_grads());
  layer_object->set_floats("w-velocities", layer.get_w_velocities());
  layer_object->set_floats("w-m1", layer.get_w_m1());
  layer_object->set_floats("w-m2", layer.get_w_m2());
  layer_object->set_numbers("w-timesteps", layer.get_w_timesteps());
  layer_object->set_floats("w-decays", layer.get_w_decays());

  layer_object->set_floats("b-values", layer.get_b_values());
  layer_object->set_floats("b-grads", layer.get_b_grads());
  layer_object->set_floats("b-velocities", layer.get_b_velocities());
  layer_object->set_floats("b-m1", layer.get_b_m1());
  layer_object->set_floats("b-m2", layer.get_b_m2());
  layer_object->set_numbers("b-timesteps", layer.get_b_timesteps());
  layer_object->set_floats("b-decays", layer.get_b_decays());

  layer_object->set_floats("rw-values", layer.get_rw_values());
  layer_object->set_floats("rw-grads", layer.get_rw_grads());
  layer_object->set_floats("rw-velocities", layer.get_rw_velocities());
  layer_object->set_floats("rw-m1", layer.get_rw_m1());
  layer_object->set_floats("rw-m2", layer.get_rw_m2());
  layer_object->set_numbers("rw-timesteps", layer.get_rw_timesteps());
  layer_object->set_floats("rw-decays", layer.get_rw_decays());

  layer_object->set_floats("z-w-values", layer.get_z_w_values());
  layer_object->set_floats("z-w-grads", layer.get_z_w_grads());
  layer_object->set_floats("z-w-velocities", layer.get_z_w_velocities());
  layer_object->set_floats("z-w-m1", layer.get_z_w_m1());
  layer_object->set_floats("z-w-m2", layer.get_z_w_m2());
  layer_object->set_numbers("z-w-timesteps", layer.get_z_w_timesteps());
  layer_object->set_floats("z-w-decays", layer.get_z_w_decays());
  
  layer_object->set_floats("z-rw-values", layer.get_z_rw_values());
  layer_object->set_floats("z-rw-grads", layer.get_z_rw_grads());
  layer_object->set_floats("z-rw-velocities", layer.get_z_rw_velocities());
  layer_object->set_floats("z-rw-m1", layer.get_z_rw_m1());
  layer_object->set_floats("z-rw-m2", layer.get_z_rw_m2());
  layer_object->set_numbers("z-rw-timesteps", layer.get_z_rw_timesteps());
  layer_object->set_floats("z-rw-decays", layer.get_z_rw_decays());

  layer_object->set_floats("z-b-values", layer.get_z_b_values());
  layer_object->set_floats("z-b-grads", layer.get_z_b_grads());
  layer_object->set_floats("z-b-velocities", layer.get_z_b_velocities());
  layer_object->set_floats("z-b-m1", layer.get_z_b_m1());
  layer_object->set_floats("z-b-m2", layer.get_z_b_m2());
  layer_object->set_numbers("z-b-timesteps", layer.get_z_b_timesteps());
  layer_object->set_floats("z-b-decays", layer.get_z_b_decays());

  // Reset Gate (r)
  layer_object->set_floats("r-w-values", layer.get_r_w_values());
  layer_object->set_floats("r-w-grads", layer.get_r_w_grads());
  layer_object->set_floats("r-w-velocities", layer.get_r_w_velocities());
  layer_object->set_floats("r-w-m1", layer.get_r_w_m1());
  layer_object->set_floats("r-w-m2", layer.get_r_w_m2());
  layer_object->set_numbers("r-w-timesteps", layer.get_r_w_timesteps());
  layer_object->set_floats("r-w-decays", layer.get_r_w_decays());

  layer_object->set_floats("r-rw-values", layer.get_r_rw_values());
  layer_object->set_floats("r-rw-grads", layer.get_r_rw_grads());
  layer_object->set_floats("r-rw-velocities", layer.get_r_rw_velocities());
  layer_object->set_floats("r-rw-m1", layer.get_r_rw_m1());
  layer_object->set_floats("r-rw-m2", layer.get_r_rw_m2());
  layer_object->set_numbers("r-rw-timesteps", layer.get_r_rw_timesteps());
  layer_object->set_floats("r-rw-decays", layer.get_r_rw_decays());

  layer_object->set_floats("r-b-values", layer.get_r_b_values());
  layer_object->set_floats("r-b-grads", layer.get_r_b_grads());
  layer_object->set_floats("r-b-velocities", layer.get_r_b_velocities());
  layer_object->set_floats("r-b-m1", layer.get_r_b_m1());
  layer_object->set_floats("r-b-m2", layer.get_r_b_m2());
  layer_object->set_numbers("r-b-timesteps", layer.get_r_b_timesteps());
  layer_object->set_floats("r-b-decays", layer.get_r_b_decays());

  auto residual_projector = add_residual_projector(layer.get_residual_projector());
  if (residual_projector != nullptr)
  {
    layer_object->set("residual-projector", residual_projector);
    delete residual_projector;
  }

  layers.add(layer_object);
  delete layer_array;
  delete layer_object;
}

void NeuralNetworkSerializer::add_fflayer(const FFLayer& layer, TinyJSON::TJValueArray& layers)
{
  auto layer_object = new TinyJSON::TJValueObject();
  auto layer_array = new TinyJSON::TJValueArray();
  for(const auto& neuron : layer.get_neurons())
  {
    auto* neuron_object = add_neuron(neuron);
    layer_array->add(neuron_object);
    delete neuron_object;
  }
  layer_object->set_string("layer-name", "fflayer");
  layer_object->set("neurons", layer_array);
  layer_object->set_number("residual-layer-number", layer.get_residual_layer_number());
  layer_object->set_string("optimiser-type", optimiser_type_to_string(layer.get_optimiser_type()).c_str());
  layer_object->set_string("activation-method", layer.get_activation().method_to_string().c_str());
  layer_object->set_float("activation-alpha", layer.get_activation().get_alpha());
  layer_object->set_number("layer-type", (int)layer.get_layer_type());

  layer_object->set_number("number-input-neurons", layer.get_number_input_neurons());
  layer_object->set_number("number-output-neurons", layer.get_number_output_neurons());
  layer_object->set_floats("w-values", layer.get_w_values());
  layer_object->set_floats("w-grads", layer.get_w_grads());
  layer_object->set_floats("w-velocities", layer.get_w_velocities());
  layer_object->set_floats("w-m1", layer.get_w_m1());
  layer_object->set_floats("w-m2", layer.get_w_m2());
  layer_object->set_numbers("w-timesteps", layer.get_w_timesteps());
  layer_object->set_floats("w-decays", layer.get_w_decays());

  layer_object->set_floats("b-values", layer.get_b_values());
  layer_object->set_floats("b-grads", layer.get_b_grads());
  layer_object->set_floats("b-velocities", layer.get_b_velocities());
  layer_object->set_floats("b-m1", layer.get_b_m1());
  layer_object->set_floats("b-m2", layer.get_b_m2());
  layer_object->set_numbers("b-timesteps", layer.get_b_timesteps());
  layer_object->set_floats("b-decays", layer.get_b_decays());

  auto residual_projector = add_residual_projector(layer.get_residual_projector());
  if (residual_projector != nullptr)
  {
    layer_object->set("residual-projector", residual_projector);
    delete residual_projector;
  }

  layers.add(layer_object);
  delete layer_array;
  delete layer_object;
}

TinyJSON::TJValueArray* NeuralNetworkSerializer::add_hidden_layers(const std::vector<LayerDetails> hidden_layers)
{
  auto hidden_layers_array = new TinyJSON::TJValueArray();
  for (const auto& hl : hidden_layers)
  {
    auto hidden_layer_object = new TinyJSON::TJValueObject();
    hidden_layer_object->set_string("type", hl.get_type_string().c_str());
    hidden_layer_object->set_number("size", hl.get_size());

    hidden_layers_array->add(hidden_layer_object);
    delete hidden_layer_object;
  }
  return hidden_layers_array;
}

TinyJSON::TJValueObject* NeuralNetworkSerializer::add_residual_projector(const ResidualProjector* residual_projector)
{
  if (nullptr == residual_projector)
  {
    return nullptr;
  }
  auto residual_projector_object = new TinyJSON::TJValueObject();
  residual_projector_object->set_number("input-size", residual_projector->get_input_size());
  residual_projector_object->set_number("output-size", residual_projector->get_output_size());
  residual_projector_object->set_floats("w-values", residual_projector->get_w_values());
  residual_projector_object->set_floats("w-grads", residual_projector->get_w_grads());
  residual_projector_object->set_floats("w-velocities", residual_projector->get_w_velocities());
  residual_projector_object->set_floats("w-m1", residual_projector->get_w_m1());
  residual_projector_object->set_floats("w-m2", residual_projector->get_w_m2());
  residual_projector_object->set_numbers("w-timesteps", residual_projector->get_w_timesteps());
  residual_projector_object->set_floats("w-decays", residual_projector->get_w_decays());
  return residual_projector_object;
}

void NeuralNetworkSerializer::add_layers(const NeuralNetwork& nn, TinyJSON::TJValueObject& json)
{
  auto layers_array = new TinyJSON::TJValueArray();
  const auto& layers = nn.get_layers();
  for(const auto& layer : layers.get_layers())
  {
    auto fflayer = dynamic_cast<FFLayer*>(layer.get());
    if (nullptr != fflayer)
    {
      add_fflayer(*fflayer, *layers_array);
      continue;
    }

    auto elmanrnnlayer = dynamic_cast<ElmanRNNLayer*>(layer.get());
    if (nullptr != elmanrnnlayer)
    {
      add_elmanrnnlayer(*elmanrnnlayer, *layers_array);
      continue;
    }

    auto grulayer = dynamic_cast<GRURNNLayer*>(layer.get());
    if (nullptr != grulayer)
    {
      add_grurnnlayer(*grulayer, *layers_array);
      continue;
    }

    Logger::panic("Unknown layer type!");
  }
  json.set("layers", layers_array);
  json.set_float("layers-weight-decay", layers.get_weight_decay());
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
    ErrorCalculation::type::bce_loss,
    ErrorCalculation::type::cross_entropy
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
  tj_errors->set_float("cross-entropy", metrics[10].error());

  json.set("errors", tj_errors);
  delete tj_errors;
}

void NeuralNetworkSerializer::add_final_learning_rate(const NeuralNetwork& nn, TinyJSON::TJValueObject& json)
{
  json.set_float("final-learning-rate", nn.get_learning_rate());
}
