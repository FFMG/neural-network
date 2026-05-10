#include <chrono>
#include <memory>

#include "./libraries/instrumentor.h"
#include "logger.h"
#include "neuralnetworkserializer.h"
#include "optimiser.h"

NeuralNetworkSerializer::NeuralNetworkSerializer()
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
}

NeuralNetwork* NeuralNetworkSerializer::load(const std::string& path)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  //  any errors will not throw, just return null
  TinyJSON::parse_options options_parse = {};
  options_parse.throw_exception = false;
  options_parse.callback_function = [&](TinyJSON::parse_options::message_type message_type, const TJCHAR* exception_message)
    {
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

  // if we have a final learning rate we want to use that.
  auto json_object = dynamic_cast<const TinyJSON::TJValueObject*>(tj);
  if (nullptr != json_object)
  {
    auto final_learning_rate = json_object->get_or<double>("final-learning-rate", 0.0);
    if (final_learning_rate != 0.0)
    {
      options.with_learning_rate(final_learning_rate);
    }
  }

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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  std::vector<std::unique_ptr<Layer>> layers = {};
  auto number_of_layers = get_number_of_layers(json);
  if(number_of_layers <= 2)
  {
    Logger::panic("The number of layers must be at least 2, (input+output)");
  }

  layers.reserve(number_of_layers);
  
  // create the hidden layers.
  auto* layers_array = get_layers_array(json);
  if (nullptr == layers_array)
  {
    Logger::panic("Could not locate the layers array.");
  }

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

    if (type == "ffoutputlayer")
    {
      layers.emplace_back(
        std::move(create_ffoutputlayer(layer_index, *layer_object, options.number_of_threads(), options.output_layer_details() ))
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

    if (type == "lstmlayer")
    {
      layers.emplace_back(
        std::move(create_lstmlayer(layer_index, *layer_object, options.number_of_threads()))
      );
      continue;
    }

    if (type == "multioutputlayer")
    {
      layers.emplace_back(
        std::move(create_multioutputlayer(layer_index, *layer_object, options.number_of_threads(), options.multi_output_layer_details()))
      );
      continue;
    }

    Logger::panic("Unknown Layer type:", type);
  }

  return Layers(options, layers);
}

std::unique_ptr<Layer> NeuralNetworkSerializer::create_elmanrnnlayer(
  unsigned layer_index,
  const TinyJSON::TJValueObject& layer_object,
  int number_of_threads
)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  // get the neurons
  auto neurons = get_neurons(layer_object, layer_index);

  auto residual_layer_number = layer_object.get<int>("residual-layer-number");
  auto optimiser_type_string = layer_object.try_get_string("optimiser-type");
  if (optimiser_type_string == nullptr)
  {
    Logger::panic("Missing layer 'optimiser-type'.");
  }
  auto optimiser_type = string_to_optimiser_type(optimiser_type_string);

  auto layer_role_number = layer_object.get<int>("layer-role");
  auto layer_role = (Layer::Role)layer_role_number;

  auto number_input_neurons = layer_object.get<unsigned>("number-input-neurons");
  auto number_output_neurons = layer_object.get<unsigned>("number-output-neurons");

  auto lah = get_activation_helper(layer_object, number_input_neurons, number_output_neurons);

  auto w_values = layer_object.get<std::vector<double>>("w-values");
  auto w_grads = layer_object.get<std::vector<double>>("w-grads");
  auto w_velocities = layer_object.get<std::vector<double>>("w-velocities");
  auto w_m1 = layer_object.get<std::vector<double>>("w-m1");
  auto w_m2 = layer_object.get<std::vector<double>>("w-m2");
  auto w_timesteps = layer_object.get<std::vector<long long>>("w-timesteps");
  auto w_decays = layer_object.get<std::vector<double>>("w-decays");

  auto b_values = layer_object.get<std::vector<double>>("b-values");
  auto b_grads = layer_object.get<std::vector<double>>("b-grads");
  auto b_velocities = layer_object.get<std::vector<double>>("b-velocities");
  auto b_m1 = layer_object.get<std::vector<double>>("b-m1");
  auto b_m2 = layer_object.get<std::vector<double>>("b-m2");
  auto b_timesteps = layer_object.get<std::vector<long long>>("b-timesteps");
  auto b_decays = layer_object.get<std::vector<double>>("b-decays");

  auto rw_values = layer_object.get<std::vector<double>>("rw-values");
  auto rw_grads = layer_object.get<std::vector<double>>("rw-grads");
  auto rw_velocities = layer_object.get<std::vector<double>>("rw-velocities");
  auto rw_m1 = layer_object.get<std::vector<double>>("rw-m1");
  auto rw_m2 = layer_object.get<std::vector<double>>("rw-m2");
  auto rw_timesteps = layer_object.get<std::vector<long long>>("rw-timesteps");
  auto rw_decays = layer_object.get<std::vector<double>>("rw-decays");

  double momentum = layer_object.get_or<double>("momentum", 0.0);

  std::unique_ptr<ResidualProjector> residual_projector(get_residual_projector(layer_object));

  auto layer = std::make_unique<ElmanRNNLayer>(
    layer_index,
    layer_role,
    optimiser_type,
    residual_layer_number,
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
    residual_projector.get(),
    number_of_threads,
    lah,
    momentum
  );

  return layer;
}

std::unique_ptr<Layer> NeuralNetworkSerializer::create_grurnnlayer(
  unsigned layer_index,
  const TinyJSON::TJValueObject& layer_object,
  int number_of_threads
)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  // get the neurons
  auto neurons = get_neurons(layer_object, layer_index);

  auto residual_layer_number = layer_object.get<int>("residual-layer-number");
  auto optimiser_type_string = layer_object.try_get_string("optimiser-type");
  if (optimiser_type_string == nullptr)
  {
    Logger::panic("Missing layer 'optimiser-type'.");
  }
  auto optimiser_type = string_to_optimiser_type(optimiser_type_string);

  auto layer_role_number = layer_object.get<int>("layer-role");
  auto layer_role = (Layer::Role)layer_role_number;

  auto number_input_neurons = layer_object.get<unsigned>("number-input-neurons");
  auto number_output_neurons = layer_object.get<unsigned>("number-output-neurons");

  auto lah = get_activation_helper(layer_object, number_input_neurons, number_output_neurons);

  auto w_values = layer_object.get<std::vector<double>>("w-values");
  auto w_grads = layer_object.get<std::vector<double>>("w-grads");
  auto w_velocities = layer_object.get<std::vector<double>>("w-velocities");
  auto w_m1 = layer_object.get<std::vector<double>>("w-m1");
  auto w_m2 = layer_object.get<std::vector<double>>("w-m2");
  auto w_timesteps = layer_object.get<std::vector<long long>>("w-timesteps");
  auto w_decays = layer_object.get<std::vector<double>>("w-decays");

  auto b_values = layer_object.get<std::vector<double>>("b-values");
  auto b_grads = layer_object.get<std::vector<double>>("b-grads");
  auto b_velocities = layer_object.get<std::vector<double>>("b-velocities");
  auto b_m1 = layer_object.get<std::vector<double>>("b-m1");
  auto b_m2 = layer_object.get<std::vector<double>>("b-m2");
  auto b_timesteps = layer_object.get<std::vector<long long>>("b-timesteps");
  auto b_decays = layer_object.get<std::vector<double>>("b-decays");

  auto rw_values = layer_object.get<std::vector<double>>("rw-values");
  auto rw_grads = layer_object.get<std::vector<double>>("rw-grads");
  auto rw_velocities = layer_object.get<std::vector<double>>("rw-velocities");
  auto rw_m1 = layer_object.get<std::vector<double>>("rw-m1");
  auto rw_m2 = layer_object.get<std::vector<double>>("rw-m2");
  auto rw_timesteps = layer_object.get<std::vector<long long>>("rw-timesteps");
  auto rw_decays = layer_object.get<std::vector<double>>("rw-decays");

  auto z_w_values = layer_object.get<std::vector<double>>("z-w-values");
  auto z_w_grads = layer_object.get<std::vector<double>>("z-w-grads");
  auto z_w_velocities = layer_object.get<std::vector<double>>("z-w-velocities");
  auto z_w_m1 = layer_object.get<std::vector<double>>("z-w-m1");
  auto z_w_m2 = layer_object.get<std::vector<double>>("z-w-m2");
  auto z_w_timesteps = layer_object.get<std::vector<long long>>("z-w-timesteps");
  auto z_w_decays = layer_object.get<std::vector<double>>("z-w-decays");

  auto z_rw_values = layer_object.get<std::vector<double>>("z-rw-values");
  auto z_rw_grads = layer_object.get<std::vector<double>>("z-rw-grads");
  auto z_rw_velocities = layer_object.get<std::vector<double>>("z-rw-velocities");
  auto z_rw_m1 = layer_object.get<std::vector<double>>("z-rw-m1");
  auto z_rw_m2 = layer_object.get<std::vector<double>>("z-rw-m2");
  auto z_rw_timesteps = layer_object.get<std::vector<long long>>("z-rw-timesteps");
  auto z_rw_decays = layer_object.get<std::vector<double>>("z-rw-decays");

  auto z_b_values = layer_object.get<std::vector<double>>("z-b-values");
  auto z_b_grads = layer_object.get<std::vector<double>>("z-b-grads");
  auto z_b_velocities = layer_object.get<std::vector<double>>("z-b-velocities");
  auto z_b_m1 = layer_object.get<std::vector<double>>("z-b-m1");
  auto z_b_m2 = layer_object.get<std::vector<double>>("z-b-m2");
  auto z_b_timesteps = layer_object.get<std::vector<long long>>("z-b-timesteps");
  auto z_b_decays = layer_object.get<std::vector<double>>("z-b-decays");

  auto r_w_values = layer_object.get<std::vector<double>>("r-w-values");
  auto r_w_grads = layer_object.get<std::vector<double>>("r-w-grads");
  auto r_w_velocities = layer_object.get<std::vector<double>>("r-w-velocities");
  auto r_w_m1 = layer_object.get<std::vector<double>>("r-w-m1");
  auto r_w_m2 = layer_object.get<std::vector<double>>("r-w-m2");
  auto r_w_timesteps = layer_object.get<std::vector<long long>>("r-w-timesteps");
  auto r_w_decays = layer_object.get<std::vector<double>>("r-w-decays");

  auto r_rw_values = layer_object.get<std::vector<double>>("r-rw-values");
  auto r_rw_grads = layer_object.get<std::vector<double>>("r-rw-grads");
  auto r_rw_velocities = layer_object.get<std::vector<double>>("r-rw-velocities");
  auto r_rw_m1 = layer_object.get<std::vector<double>>("r-rw-m1");
  auto r_rw_m2 = layer_object.get<std::vector<double>>("r-rw-m2");
  auto r_rw_timesteps = layer_object.get<std::vector<long long>>("r-rw-timesteps");
  auto r_rw_decays = layer_object.get<std::vector<double>>("r-rw-decays");

  auto r_b_values = layer_object.get<std::vector<double>>("r-b-values");
  auto r_b_grads = layer_object.get<std::vector<double>>("r-b-grads");
  auto r_b_velocities = layer_object.get<std::vector<double>>("r-b-velocities");
  auto r_b_m1 = layer_object.get<std::vector<double>>("r-b-m1");
  auto r_b_m2 = layer_object.get<std::vector<double>>("r-b-m2");
  auto r_b_timesteps = layer_object.get<std::vector<long long>>("r-b-timesteps");
  auto r_b_decays = layer_object.get<std::vector<double>>("r-b-decays");

  double momentum = layer_object.get_or<double>("momentum", 0.0);

  std::unique_ptr<ResidualProjector> residual_projector(get_residual_projector(layer_object));

  auto layer = std::make_unique<GRURNNLayer>(
    layer_index,
    layer_role,
    optimiser_type,
    residual_layer_number,
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
    residual_projector.get(),
    number_of_threads,
    lah,
    momentum
  );

  return layer;
}

std::unique_ptr<Layer> NeuralNetworkSerializer::create_lstmlayer(
  unsigned layer_index,
  const TinyJSON::TJValueObject& layer_object,
  int number_of_threads
)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  // get the neurons
  auto neurons = get_neurons(layer_object, layer_index);

  auto residual_layer_number = layer_object.get<int>("residual-layer-number");
  auto optimiser_type_string = layer_object.try_get_string("optimiser-type");
  if (optimiser_type_string == nullptr)
  {
    Logger::panic("Missing layer 'optimiser-type'.");
  }
  auto optimiser_type = string_to_optimiser_type(optimiser_type_string);

  auto layer_role_number = layer_object.get<int>("layer-role");
  auto layer_role = (Layer::Role)layer_role_number;

  auto number_input_neurons = layer_object.get<unsigned>("number-input-neurons");
  auto number_output_neurons = layer_object.get<unsigned>("number-output-neurons");

  auto lah = get_activation_helper(layer_object, number_input_neurons, number_output_neurons);

  // Cell Candidate weights
  auto w_values = layer_object.get<std::vector<double>>("w-values");
  auto w_grads = layer_object.get<std::vector<double>>("w-grads");
  auto w_velocities = layer_object.get<std::vector<double>>("w-velocities");
  auto w_m1 = layer_object.get<std::vector<double>>("w-m1");
  auto w_m2 = layer_object.get<std::vector<double>>("w-m2");
  auto w_timesteps = layer_object.get<std::vector<long long>>("w-timesteps");
  auto w_decays = layer_object.get<std::vector<double>>("w-decays");

  auto b_values = layer_object.get<std::vector<double>>("b-values");
  auto b_grads = layer_object.get<std::vector<double>>("b-grads");
  auto b_velocities = layer_object.get<std::vector<double>>("b-velocities");
  auto b_m1 = layer_object.get<std::vector<double>>("b-m1");
  auto b_m2 = layer_object.get<std::vector<double>>("b-m2");
  auto b_timesteps = layer_object.get<std::vector<long long>>("b-timesteps");
  auto b_decays = layer_object.get<std::vector<double>>("b-decays");

  auto rw_values = layer_object.get<std::vector<double>>("rw-values");
  auto rw_grads = layer_object.get<std::vector<double>>("rw-grads");
  auto rw_velocities = layer_object.get<std::vector<double>>("rw-velocities");
  auto rw_m1 = layer_object.get<std::vector<double>>("rw-m1");
  auto rw_m2 = layer_object.get<std::vector<double>>("rw-m2");
  auto rw_timesteps = layer_object.get<std::vector<long long>>("rw-timesteps");
  auto rw_decays = layer_object.get<std::vector<double>>("rw-decays");

  // Forget Gate
  auto f_w_values = layer_object.get<std::vector<double>>("f-w-values");
  auto f_w_grads = layer_object.get<std::vector<double>>("f-w-grads");
  auto f_w_velocities = layer_object.get<std::vector<double>>("f-w-velocities");
  auto f_w_m1 = layer_object.get<std::vector<double>>("f-w-m1");
  auto f_w_m2 = layer_object.get<std::vector<double>>("f-w-m2");
  auto f_w_timesteps = layer_object.get<std::vector<long long>>("f-w-timesteps");
  auto f_w_decays = layer_object.get<std::vector<double>>("f-w-decays");

  auto f_rw_values = layer_object.get<std::vector<double>>("f-rw-values");
  auto f_rw_grads = layer_object.get<std::vector<double>>("f-rw-grads");
  auto f_rw_velocities = layer_object.get<std::vector<double>>("f-rw-velocities");
  auto f_rw_m1 = layer_object.get<std::vector<double>>("f-rw-m1");
  auto f_rw_m2 = layer_object.get<std::vector<double>>("f-rw-m2");
  auto f_rw_timesteps = layer_object.get<std::vector<long long>>("f-rw-timesteps");
  auto f_rw_decays = layer_object.get<std::vector<double>>("f-rw-decays");

  auto f_b_values = layer_object.get<std::vector<double>>("f-b-values");
  auto f_b_grads = layer_object.get<std::vector<double>>("f-b-grads");
  auto f_b_velocities = layer_object.get<std::vector<double>>("f-b-velocities");
  auto f_b_m1 = layer_object.get<std::vector<double>>("f-b-m1");
  auto f_b_m2 = layer_object.get<std::vector<double>>("f-b-m2");
  auto f_b_timesteps = layer_object.get<std::vector<long long>>("f-b-timesteps");
  auto f_b_decays = layer_object.get<std::vector<double>>("f-b-decays");

  // Input Gate
  auto i_w_values = layer_object.get<std::vector<double>>("i-w-values");
  auto i_w_grads = layer_object.get<std::vector<double>>("i-w-grads");
  auto i_w_velocities = layer_object.get<std::vector<double>>("i-w-velocities");
  auto i_w_m1 = layer_object.get<std::vector<double>>("i-w-m1");
  auto i_w_m2 = layer_object.get<std::vector<double>>("i-w-m2");
  auto i_w_timesteps = layer_object.get<std::vector<long long>>("i-w-timesteps");
  auto i_w_decays = layer_object.get<std::vector<double>>("i-w-decays");

  auto i_rw_values = layer_object.get<std::vector<double>>("i-rw-values");
  auto i_rw_grads = layer_object.get<std::vector<double>>("i-rw-grads");
  auto i_rw_velocities = layer_object.get<std::vector<double>>("i-rw-velocities");
  auto i_rw_m1 = layer_object.get<std::vector<double>>("i-rw-m1");
  auto i_rw_m2 = layer_object.get<std::vector<double>>("i-rw-m2");
  auto i_rw_timesteps = layer_object.get<std::vector<long long>>("i-rw-timesteps");
  auto i_rw_decays = layer_object.get<std::vector<double>>("i-rw-decays");

  auto i_b_values = layer_object.get<std::vector<double>>("i-b-values");
  auto i_b_grads = layer_object.get<std::vector<double>>("i-b-grads");
  auto i_b_velocities = layer_object.get<std::vector<double>>("i-b-velocities");
  auto i_b_m1 = layer_object.get<std::vector<double>>("i-b-m1");
  auto i_b_m2 = layer_object.get<std::vector<double>>("i-b-m2");
  auto i_b_timesteps = layer_object.get<std::vector<long long>>("i-b-timesteps");
  auto i_b_decays = layer_object.get<std::vector<double>>("i-b-decays");

  // Output Gate
  auto o_w_values = layer_object.get<std::vector<double>>("o-w-values");
  auto o_w_grads = layer_object.get<std::vector<double>>("o-w-grads");
  auto o_w_velocities = layer_object.get<std::vector<double>>("o-w-velocities");
  auto o_w_m1 = layer_object.get<std::vector<double>>("o-w-m1");
  auto o_w_m2 = layer_object.get<std::vector<double>>("o-w-m2");
  auto o_w_timesteps = layer_object.get<std::vector<long long>>("o-w-timesteps");
  auto o_w_decays = layer_object.get<std::vector<double>>("o-w-decays");

  auto o_rw_values = layer_object.get<std::vector<double>>("o-rw-values");
  auto o_rw_grads = layer_object.get<std::vector<double>>("o-rw-grads");
  auto o_rw_velocities = layer_object.get<std::vector<double>>("o-rw-velocities");
  auto o_rw_m1 = layer_object.get<std::vector<double>>("o-rw-m1");
  auto o_rw_m2 = layer_object.get<std::vector<double>>("o-rw-m2");
  auto o_rw_timesteps = layer_object.get<std::vector<long long>>("o-rw-timesteps");
  auto o_rw_decays = layer_object.get<std::vector<double>>("o-rw-decays");

  auto o_b_values = layer_object.get<std::vector<double>>("o-b-values");
  auto o_b_grads = layer_object.get<std::vector<double>>("o-b-grads");
  auto o_b_velocities = layer_object.get<std::vector<double>>("o-b-velocities");
  auto o_b_m1 = layer_object.get<std::vector<double>>("o-b-m1");
  auto o_b_m2 = layer_object.get<std::vector<double>>("o-b-m2");
  auto o_b_timesteps = layer_object.get<std::vector<long long>>("o-b-timesteps");
  auto o_b_decays = layer_object.get<std::vector<double>>("o-b-decays");

  double momentum = layer_object.get_or<double>("momentum", 0.0);

  std::unique_ptr<ResidualProjector> residual_projector(get_residual_projector(layer_object));

  auto layer = std::make_unique<LSTMLayer>(
    layer_index,
    layer_role,
    optimiser_type,
    residual_layer_number,
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
    // Forget Gate
    f_w_values, f_w_grads, f_w_velocities, f_w_m1, f_w_m2, f_w_timesteps, f_w_decays,
    f_rw_values, f_rw_grads, f_rw_velocities, f_rw_m1, f_rw_m2, f_rw_timesteps, f_rw_decays,
    f_b_values, f_b_grads, f_b_velocities, f_b_m1, f_b_m2, f_b_timesteps, f_b_decays,
    // Input Gate
    i_w_values, i_w_grads, i_w_velocities, i_w_m1, i_w_m2, i_w_timesteps, i_w_decays,
    i_rw_values, i_rw_grads, i_rw_velocities, i_rw_m1, i_rw_m2, i_rw_timesteps, i_rw_decays,
    i_b_values, i_b_grads, i_b_velocities, i_b_m1, i_b_m2, i_b_timesteps, i_b_decays,
    // Output Gate
    o_w_values, o_w_grads, o_w_velocities, o_w_m1, o_w_m2, o_w_timesteps, o_w_decays,
    o_rw_values, o_rw_grads, o_rw_velocities, o_rw_m1, o_rw_m2, o_rw_timesteps, o_rw_decays,
    o_b_values, o_b_grads, o_b_velocities, o_b_m1, o_b_m2, o_b_timesteps, o_b_decays,
    residual_projector.get(),
    number_of_threads,
    lah,
    momentum
  );

  return layer;
}

std::unique_ptr<Layer> NeuralNetworkSerializer::create_fflayer(
  unsigned layer_index,
  const TinyJSON::TJValueObject& layer_object,
  int number_of_threads
)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  // get the neurons
  auto neurons = get_neurons(layer_object, layer_index);

  auto residual_layer_number = layer_object.get<int>("residual-layer-number");
  auto optimiser_type_string = layer_object.try_get_string("optimiser-type");
  if (optimiser_type_string == nullptr)
  {
    Logger::panic("Missing layer 'optimiser-type'.");
  }
  auto optimiser_type = string_to_optimiser_type(optimiser_type_string);

  auto layer_role_number = layer_object.get<int>("layer-role");
  auto layer_role = (Layer::Role)layer_role_number;

  auto number_input_neurons = layer_object.get<unsigned>("number-input-neurons");
  auto number_output_neurons = layer_object.get<unsigned>("number-output-neurons");

  auto lah = get_activation_helper(layer_object, number_input_neurons, number_output_neurons);

  auto w_values = layer_object.get<std::vector<double>>("w-values");
  auto w_grads = layer_object.get<std::vector<double>>("w-grads");
  auto w_velocities = layer_object.get<std::vector<double>>("w-velocities");
  auto w_m1 = layer_object.get<std::vector<double>>("w-m1");
  auto w_m2 = layer_object.get<std::vector<double>>("w-m2");
  auto w_timesteps = layer_object.get<std::vector<long long>>("w-timesteps");
  auto w_decays = layer_object.get<std::vector<double>>("w-decays");
  auto b_values = layer_object.get<std::vector<double>>("b-values");
  auto b_grads = layer_object.get<std::vector<double>>("b-grads");
  auto b_velocities = layer_object.get<std::vector<double>>("b-velocities");
  auto b_m1 = layer_object.get<std::vector<double>>("b-m1");
  auto b_m2 = layer_object.get<std::vector<double>>("b-m2");
  auto b_timesteps = layer_object.get<std::vector<long long>>("b-timesteps");
  auto b_decays = layer_object.get<std::vector<double>>("b-decays");
  double momentum = layer_object.get_or<double>("momentum", 0.0);

  std::unique_ptr<ResidualProjector> residual_projector(get_residual_projector(layer_object));

  auto layer = std::make_unique<FFLayer>(
    layer_index,
    layer_role,
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
    residual_projector.get(),
    number_of_threads,
    lah,
    momentum
  );

  return layer;
}

std::unique_ptr<Layer> NeuralNetworkSerializer::create_ffoutputlayer(
  unsigned layer_index,
  const TinyJSON::TJValueObject& layer_object,
  int number_of_threads,
  const std::vector<OutputLayerDetails>& output_layer_details
)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  // get the neurons
  auto neurons = get_neurons(layer_object, layer_index);

  auto number_input_neurons = layer_object.get<unsigned>("number-input-neurons");
  auto number_output_neurons = layer_object.get<unsigned>("number-output-neurons");
  auto w_values = layer_object.get<std::vector<double>>("w-values");
  auto w_grads = layer_object.get<std::vector<double>>("w-grads");
  auto w_velocities = layer_object.get<std::vector<double>>("w-velocities");
  auto w_m1 = layer_object.get<std::vector<double>>("w-m1");
  auto w_m2 = layer_object.get<std::vector<double>>("w-m2");
  auto w_timesteps = layer_object.get<std::vector<long long>>("w-timesteps");
  auto w_decays = layer_object.get<std::vector<double>>("w-decays");
  auto b_values = layer_object.get<std::vector<double>>("b-values");
  auto b_grads = layer_object.get<std::vector<double>>("b-grads");
  auto b_velocities = layer_object.get<std::vector<double>>("b-velocities");
  auto b_m1 = layer_object.get<std::vector<double>>("b-m1");
  auto b_m2 = layer_object.get<std::vector<double>>("b-m2");
  auto b_timesteps = layer_object.get<std::vector<long long>>("b-timesteps");
  auto b_decays = layer_object.get<std::vector<double>>("b-decays");

  auto layer = std::make_unique<FFOutputLayer>(
    layer_index,
    output_layer_details,
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
    number_of_threads
  );

  return layer;
}

std::vector<OutputLayerDetails> NeuralNetworkSerializer::get_output_layer_details(const TinyJSON::TJValueObject& options_object)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  const auto* output_layer_array = static_cast<const TinyJSON::TJValueArray*>(options_object.try_get_value("output-layer-details"));
  if (nullptr == output_layer_array)
  {
    Logger::panic("Could not find output layer details option!");
  }

  std::vector<OutputLayerDetails> details;
  for (const auto& output_layer_value : *output_layer_array)
  {
    const auto* output_layer_object = static_cast<const TinyJSON::TJValueObject*>(&output_layer_value);
    if (nullptr == output_layer_object)
    {
      Logger::panic("One or more of the output layer details is not a valid object!");
    }

    auto output_error_calculation_typ_str = output_layer_object->try_get_string("error-calculation-type", true);
    if (nullptr == output_error_calculation_typ_str)
    {
      Logger::panic("Could not find output layer error-calculation-type option!");
    }
    const auto output_method = activation::string_to_method(output_layer_object->try_get_string("activation-method", false));
    const auto output_alpha = (double)output_layer_object->get_float("activation-alpha");
    const auto output_temperature = output_layer_object->get_or<double>("activation-temperature", 1.0);
    const auto output_inference_temperature = output_layer_object->get_or<double>("activation-inference-temperature", output_temperature);
    const auto output_error_calculation_type = ErrorCalculation::string_to_type(output_error_calculation_typ_str);
    const auto output_activation_method = activation(output_method, output_alpha, output_temperature, output_inference_temperature);
    const auto optimiser_type = string_to_optimiser_type(output_layer_object->try_get_string("optimiser-type", false));
    const auto momentum = output_layer_object->get<double>("momentum");

    const auto error_evaluation_config = get_error_evaluation_config(output_layer_object);
    details.push_back(OutputLayerDetails(
      output_layer_object->get<unsigned>("size"),
      output_activation_method, 
      output_error_calculation_type, 
      error_evaluation_config,
      output_layer_object->get<double>("weight-decay"),
      optimiser_type,
      momentum
    ));
  }
  return details;
}

EvaluationConfig NeuralNetworkSerializer::get_error_evaluation_config(const TinyJSON::TJValueObject* parent)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  if (nullptr == parent)
  {
    Logger::panic("Missing Rules `error-evaluation-config`.");
  }

  auto error_evaluation_config_object = static_cast<const TinyJSON::TJValueObject*>(parent->try_get_value("error-evaluation-config", true));
  
  if (nullptr == error_evaluation_config_object)
  {
    Logger::panic("Could not locate the error-evaluation-config section!");
  }

  return EvaluationConfig(
    error_evaluation_config_object->get<double>("neutral-tolerance"),
    error_evaluation_config_object->get<double>("confidence-threshold"),
    error_evaluation_config_object->get<double>("huber-delta"),
    error_evaluation_config_object->get<double>("direction-lambda"),
    error_evaluation_config_object->get_boolean("use-direction-penalty"),
    error_evaluation_config_object->get<double>("cross-entropy-lambda")
        );
}

std::vector<LayerDetails> NeuralNetworkSerializer::get_hidden_layers(const TinyJSON::TJValueObject& options_object)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
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
    const auto hidden_method_string = phlo->try_get_string("activation-method", false);
    const auto hidden_method = activation::string_to_method(hidden_method_string == nullptr ? "sigmoid" : hidden_method_string);
    const auto hidden_alpha = phlo->get<double>("activation-alpha");
    const auto hidden_temperature = phlo->get_or<double>("activation-temperature", 1.0);
    const auto hidden_inference_temperature = phlo->get_or<double>("activation-inference-temperature", hidden_temperature);

    const auto optimiser_type = string_to_optimiser_type(phlo->try_get_string("optimiser-type", false));
    const auto momentum = phlo->get<double>("momentum");


    const auto layer_architecture_string = phlo->try_get_string("architecture");
    hidden_layer.emplace_back(LayerDetails(
      Layer::architecture_from_string(layer_architecture_string == nullptr ? "ff" : layer_architecture_string),
      phlo->get<unsigned>("size"),
      activation(hidden_method, hidden_alpha, hidden_temperature, hidden_inference_temperature),
      phlo->get<double>("dropout"),
      phlo->get<double>("weight-decay"),
      optimiser_type,
      momentum
    ));
  }
  return hidden_layer;
}

ResidualProjector* NeuralNetworkSerializer::get_residual_projector(const TinyJSON::TJValueObject& layer_object)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  const auto* residual_projector_object = static_cast<const TinyJSON::TJValueObject*>(layer_object.try_get_value("residual-projector"));
  if (nullptr == residual_projector_object)
  {
    return nullptr;
  }

  auto input_size = residual_projector_object->get<unsigned>("input-size");
  auto output_size = residual_projector_object->get<unsigned>("output-size");
  auto w_values = residual_projector_object->get<std::vector<double>>("w-values");
  auto w_grads = residual_projector_object->get<std::vector<double>>("w-grads");
  auto w_velocities = residual_projector_object->get<std::vector<double>>("w-velocities");
  auto w_m1 = residual_projector_object->get<std::vector<double>>("w-m1");
  auto w_m2 = residual_projector_object->get<std::vector<double>>("w-m2");
  auto w_timesteps = residual_projector_object->get<std::vector<long long>>("w-timesteps");
  auto w_decays = residual_projector_object->get<std::vector<double>>("w-decays");

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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
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

NeuralNetworkOptions NeuralNetworkSerializer::get_and_build_options(const TinyJSON::TJValue& json)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
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
  
  const auto log_level_string = options_object->try_get_string("log-level", false);
  const auto log_level = Logger::string_to_level(log_level_string == nullptr ? "none" : log_level_string);
  
  const auto learning_rate = options_object->get_or<double>("learning-rate", 0.001);
  const auto learning_rate_warmup_start = options_object->get_or<double>("learning-rate-warmup-start", 0.0);
  const auto learning_rate_warmup_target = options_object->get_or<double>("learning-rate-warmup-target", 0.0);

  const auto number_of_epoch = options_object->get<int>("number-of-epoch");
  const auto batch_size = options_object->get<int>("batch-size");
  const auto data_is_unique = options_object->get<bool>("data-is-unique");
  const auto number_of_threads = options_object->get_or<int>("number-of-threads", 0);
  const auto learning_rate_decay_rate = options_object->get_or<double>("learning-rate-decay-rate", 0.0);
  const auto adaptive_learning_rate = options_object->get<bool>("adaptive-learning-rate");
  const auto optimiser_type_string = options_object->try_get_string("optimiser-type");
  (void)optimiser_type_string;

  const auto learning_rate_restart_rate = options_object->get_or<double>("learning-rate-restart-rate", 0.0);
  const auto learning_rate_restart_boost = options_object->get_or<double>("learning-rate-restart-boost", 0.0);
  const auto residual_layer_jump = options_object->get<int>("residual-layer-jump");
  const auto clip_threshold = options_object->get_or<double>("clip-threshold", 1.0);
  const auto shuffle_training_data = options_object->get<bool>("shuffle-training-data");
  const auto hidden_layers = get_hidden_layers(*options_object);
  const auto output_layer_details = get_output_layer_details(*options_object);
  const auto multi_output_layer_details = get_multi_output_layer_details(*options_object);
  
  const auto enable_bptt = options_object->get<bool>("enable-bptt");
  const auto bptt_max_ticks = options_object->get_or<int>("bptt-max-ticks", 0);
  const auto shuffle_bptt_batches = options_object->get<bool>("shuffle-bptt-batches");
  const auto has_bias = options_object->get<bool>("has-bias");

  const auto output_error_calculation_type_string = options_object->try_get_string("output-error-calculation-type");
  (void)output_error_calculation_type_string;

  const auto update_training_monitor_percent = options_object->get_or<double>("update-training-monitor-percent", 0.0);

  const auto final_error_calculation_types_array = dynamic_cast<const TinyJSON::TJValueArray*>(options_object->try_get_value("final-error-calculation-types"));
  std::vector<ErrorCalculation::type> final_error_calculation_types = {};
  if (nullptr != final_error_calculation_types_array)
  {
    for (unsigned i = 0; i < final_error_calculation_types_array->get_number_of_items(); ++i)
    {
      auto type_string = dynamic_cast<const TinyJSON::TJValueString*>(final_error_calculation_types_array->at(i));
      if (nullptr != type_string)
      {
        final_error_calculation_types.push_back(ErrorCalculation::string_to_type(type_string->get_string()));
      }
    }
  }

  auto option = NeuralNetworkOptions::create(topology)
    .with_learning_rate(learning_rate)
    .with_number_of_epoch(number_of_epoch)
    .with_batch_size(batch_size)
    .with_data_is_unique(data_is_unique)
    .with_number_of_threads(number_of_threads)
    .with_learning_rate_decay_rate(learning_rate_decay_rate)
    .with_adaptive_learning_rates(adaptive_learning_rate)
    .with_learning_rate_boost_rate(learning_rate_restart_rate, learning_rate_restart_boost)
    .with_residual_layer_jump(residual_layer_jump)
    .with_clip_threshold(clip_threshold)
    .with_learning_rate_warmup(learning_rate_warmup_start, learning_rate_warmup_target)
    .with_log_level(log_level)
    .with_shuffle_training_data(shuffle_training_data)
    .with_hidden_layers(hidden_layers)
    .with_bptt_max_ticks(bptt_max_ticks)
    .with_shuffle_bptt_batches(shuffle_bptt_batches)
    .with_final_error_calculation_types(final_error_calculation_types)
    .with_enable_bptt(enable_bptt)
    .with_update_training_monitor_percent(update_training_monitor_percent)
    .with_has_bias(has_bias);

  if (multi_output_layer_details.size())
  {
    option.with_output_layer_details(multi_output_layer_details);
  }
  else
  {
    option.with_output_layer_details(output_layer_details);
  }
  return option.build();
}

std::vector<std::map<ErrorCalculation::type, double>> NeuralNetworkSerializer::get_errors(const TinyJSON::TJValue& json)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  std::vector<std::map<ErrorCalculation::type, double>> errors;
  auto tj_object = dynamic_cast<const TinyJSON::TJValueObject*>(&json);
  if (nullptr == tj_object)
  {
    Logger::info("Could not load any errors.");
    return errors;
  }
  auto tj_errors_array = dynamic_cast<const TinyJSON::TJValueArray*>(tj_object->try_get_value("errors"));
  if (nullptr == tj_errors_array)
  {
    Logger::info("Could not load any errors.");
    return errors;
  }

  const auto& error_types = all_error_types();
  unsigned output_layer = 0;
  for (const auto& tj_errors_value : *tj_errors_array)
  {
    auto tj_errors_object = dynamic_cast<const TinyJSON::TJValueObject*>(&tj_errors_value);
    errors.push_back({});
    for (const auto& error_type : error_types)
    {
      errors[output_layer][error_type] = tj_errors_object->get<double>(ErrorCalculation::type_to_string(error_type).c_str());
    }
    ++output_layer;
  }
  return errors;
}

const TinyJSON::TJValueArray* NeuralNetworkSerializer::get_layers_array(const TinyJSON::TJValue& json)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
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

std::vector<Neuron> NeuralNetworkSerializer::get_neurons(const TinyJSON::TJValue& json, unsigned layer_number)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  const auto* layer_object = get_layer_object(json, layer_number);
  if (nullptr == layer_object)
  {
    return {};
  }
  return get_neurons(*layer_object, layer_number);
}

std::vector<Neuron> NeuralNetworkSerializer::get_neurons(const TinyJSON::TJValueObject& layer_object, unsigned layer_number)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
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

    const auto neuron_type = static_cast<Neuron::Type>(neuron_object->get<int>("neuron-type"));
    const auto dropout_rate = neuron_object->get<double>("dropout-rate");
    
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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  auto* array = get_layers_array(json);
  if(nullptr == array)
  {
    return 0;
  }
  return array->get_number_of_items();
}

layer_activation_helper NeuralNetworkSerializer::get_activation_helper(const TinyJSON::TJValueObject& layer_object, unsigned num_inputs, unsigned num_outputs)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  
  const auto method_str = layer_object.try_get_string("activation-method");
  const auto method = activation::string_to_method(method_str == nullptr ? "sigmoid" : method_str);
  const auto alpha = layer_object.get_or<double>("activation-alpha", 1.0);
  const auto temperature = layer_object.get_or<double>("activation-temperature", 1.0);
  const auto inference_temperature = layer_object.get_or<double>("activation-inference-temperature", temperature);

  activation default_activation(method, alpha, temperature, inference_temperature);

  layer_activation_helper lah(default_activation, num_inputs, num_outputs);

  auto* ranges_array = dynamic_cast<const TinyJSON::TJValueArray*>(layer_object.try_get_value("activation-ranges"));
  if (ranges_array != nullptr)
  {
    for (const auto& r_val : *ranges_array)
    {
      auto* r_obj = dynamic_cast<const TinyJSON::TJValueObject*>(&r_val);
      if (r_obj != nullptr)
      {
        const auto r_start = r_obj->get<unsigned>("start");
        const auto r_end = r_obj->get<unsigned>("end");
        const auto r_method_str = r_obj->get_string("activation-method");
        const auto r_alpha = (double)r_obj->get_float("activation-alpha");
        const auto r_temperature = r_obj->get_or<double>("activation-temperature", 1.0);
        const auto r_inference_temperature = r_obj->get_or<double>("activation-inference-temperature", r_temperature);
        lah.set_bounds(activation(activation::string_to_method(r_method_str), r_alpha, r_temperature, r_inference_temperature), r_start, r_end);
      }
    }
  }
  return lah;
}

std::vector<WeightParam> NeuralNetworkSerializer::get_weight_params(const TinyJSON::TJValueObject& parent)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  auto options_object = new TinyJSON::TJValueObject();
  
  auto topology_list = new TinyJSON::TJValueArray();
  topology_list->add_numbers(options.topology());

  auto output_layer_array = add_output_layer_details(options.output_layer_details());
  auto hidden_layer_list = add_hidden_layers(options.hidden_layers());
  
  auto multi_output_layer_details_array = new TinyJSON::TJValueArray();
  for (const auto& multi_output_layer_detail : options.multi_output_layer_details())
  {
    auto* multi_output_layer_details_obj = new TinyJSON::TJValueObject();
    auto* hidden_layers_array = add_hidden_layers(multi_output_layer_detail.get_hidden_layers());
    multi_output_layer_details_obj->set("hidden-layers", hidden_layers_array);
    delete hidden_layers_array;
    
    std::vector<OutputLayerDetails> olds = { multi_output_layer_detail.get_output_details() };
    auto* olds_arr = add_output_layer_details(olds);
    multi_output_layer_details_obj->set("output-detail", olds_arr->at(0));
    delete olds_arr;
    
    multi_output_layer_details_array->add(multi_output_layer_details_obj);
    delete multi_output_layer_details_obj;
  }
  options_object->set("multi-output-layers", multi_output_layer_details_array);
  delete multi_output_layer_details_array;

  options_object->set("topology", topology_list);
  options_object->set("output-layer-details", output_layer_array);
  options_object->set("hidden-layers", hidden_layer_list);
  options_object->set_string("log-level", Logger::level_to_string(options.log_level()).c_str());
  options_object->set_float("learning-rate", options.learning_rate());
  options_object->set_float("learning-rate-warmup-start", options.learning_rate_warmup_start());
  options_object->set_float("learning-rate-warmup-target", options.learning_rate_warmup_target());
  options_object->set_number("number-of-epoch", options.number_of_epoch());
  options_object->set_number("batch-size", options.batch_size());
  options_object->set_boolean("data-is-unique", options.data_is_unique());
  options_object->set_number("number-of-threads", options.number_of_threads());
  options_object->set_float("learning-rate-decay-rate", options.learning_rate_decay_rate());
  options_object->set_boolean("adaptive-learning-rate", options.adaptive_learning_rate());
  options_object->set_float("learning-rate-restart-rate", options.learning_rate_restart_rate());
  options_object->set_float("learning-rate-restart-boost", options.learning_rate_restart_boost());
  options_object->set_number("residual-layer-jump", options.residual_layer_jump());
  options_object->set_float("clip-threshold", options.clip_threshold());
  options_object->set_boolean("shuffle-training-data", options.shuffle_training_data());
  options_object->set_number("bptt-max-ticks", options.bptt_max_ticks());

  auto final_error_calculation_types_list = new TinyJSON::TJValueArray();
  for (const auto& type : options.final_error_calculation_types())
  {
    final_error_calculation_types_list->add_string(ErrorCalculation::type_to_string(type).c_str());
  }
  options_object->set("final-error-calculation-types", final_error_calculation_types_list);
  delete final_error_calculation_types_list;

  options_object->set_boolean("enable-bptt", options.enable_bptt());
  options_object->set_boolean("shuffle-bptt-batches", options.shuffle_bptt_batches());
  options_object->set_float("update-training-monitor-percent", options.update_training_monitor_percent());
  options_object->set_boolean("has-bias", options.has_bias());

  json.set("options", options_object);

  delete output_layer_array;
  delete hidden_layer_list;
  delete topology_list;
  delete options_object;
}

void NeuralNetworkSerializer::add_basic(TinyJSON::TJValueObject& json)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  auto now = std::chrono::system_clock::now();
  auto now_seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
  long long current_timestamp = now_seconds.time_since_epoch().count();

  json.set_number("created", current_timestamp);
}

TinyJSON::TJValueObject* NeuralNetworkSerializer::add_neuron(const Neuron& neuron)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
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
  add_activation_helper(layer.get_activation_helper(), *layer_object);
  layer_object->set_number("layer-role", (int)layer.get_layer_role());

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

  layer_object->set_float("momentum", layer.get_momentum());

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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
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
  add_activation_helper(layer.get_activation_helper(), *layer_object);
  layer_object->set_number("layer-role", (int)layer.get_layer_role());

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

  layer_object->set_float("momentum", layer.get_momentum());

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

void NeuralNetworkSerializer::add_lstmlayer(const LSTMLayer& layer, TinyJSON::TJValueArray& layers)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  auto layer_object = new TinyJSON::TJValueObject();
  auto layer_array = new TinyJSON::TJValueArray();
  for (const auto& neuron : layer.get_neurons())
  {
    auto* neuron_object = add_neuron(neuron);
    layer_array->add(neuron_object);
    delete neuron_object;
  }
  layer_object->set_string("layer-name", "lstmlayer");
  layer_object->set("neurons", layer_array);
  layer_object->set_number("residual-layer-number", layer.get_residual_layer_number());
  layer_object->set_string("optimiser-type", optimiser_type_to_string(layer.get_optimiser_type()).c_str());
  add_activation_helper(layer.get_activation_helper(), *layer_object);
  layer_object->set_number("layer-role", (int)layer.get_layer_role());

  layer_object->set_number("number-input-neurons", layer.get_number_input_neurons());
  layer_object->set_number("number-output-neurons", layer.get_number_output_neurons());
  
  // Candidate Gate (g) - uses standard w/rw/b
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

  // Forget Gate (f)
  layer_object->set_floats("f-w-values", layer.get_f_w_values());
  layer_object->set_floats("f-w-grads", layer.get_f_w_grads());
  layer_object->set_floats("f-w-velocities", layer.get_f_w_velocities());
  layer_object->set_floats("f-w-m1", layer.get_f_w_m1());
  layer_object->set_floats("f-w-m2", layer.get_f_w_m2());
  layer_object->set_numbers("f-w-timesteps", layer.get_f_w_timesteps());
  layer_object->set_floats("f-w-decays", layer.get_f_w_decays());

  layer_object->set_floats("f-rw-values", layer.get_f_rw_values());
  layer_object->set_floats("f-rw-grads", layer.get_f_rw_grads());
  layer_object->set_floats("f-rw-velocities", layer.get_f_rw_velocities());
  layer_object->set_floats("f-rw-m1", layer.get_f_rw_m1());
  layer_object->set_floats("f-rw-m2", layer.get_f_rw_m2());
  layer_object->set_numbers("f-rw-timesteps", layer.get_f_rw_timesteps());
  layer_object->set_floats("f-rw-decays", layer.get_f_rw_decays());

  layer_object->set_floats("f-b-values", layer.get_f_b_values());
  layer_object->set_floats("f-b-grads", layer.get_f_b_grads());
  layer_object->set_floats("f-b-velocities", layer.get_f_b_velocities());
  layer_object->set_floats("f-b-m1", layer.get_f_b_m1());
  layer_object->set_floats("f-b-m2", layer.get_f_b_m2());
  layer_object->set_numbers("f-b-timesteps", layer.get_f_b_timesteps());
  layer_object->set_floats("f-b-decays", layer.get_f_b_decays());

  // Input Gate (i)
  layer_object->set_floats("i-w-values", layer.get_i_w_values());
  layer_object->set_floats("i-w-grads", layer.get_i_w_grads());
  layer_object->set_floats("i-w-velocities", layer.get_i_w_velocities());
  layer_object->set_floats("i-w-m1", layer.get_i_w_m1());
  layer_object->set_floats("i-w-m2", layer.get_i_w_m2());
  layer_object->set_numbers("i-w-timesteps", layer.get_i_w_timesteps());
  layer_object->set_floats("i-w-decays", layer.get_i_w_decays());

  layer_object->set_floats("i-rw-values", layer.get_i_rw_values());
  layer_object->set_floats("i-rw-grads", layer.get_i_rw_grads());
  layer_object->set_floats("i-rw-velocities", layer.get_i_rw_velocities());
  layer_object->set_floats("i-rw-m1", layer.get_i_rw_m1());
  layer_object->set_floats("i-rw-m2", layer.get_i_rw_m2());
  layer_object->set_numbers("i-rw-timesteps", layer.get_i_rw_timesteps());
  layer_object->set_floats("i-rw-decays", layer.get_i_rw_decays());

  layer_object->set_floats("i-b-values", layer.get_i_b_values());
  layer_object->set_floats("i-b-grads", layer.get_i_b_grads());
  layer_object->set_floats("i-b-velocities", layer.get_i_b_velocities());
  layer_object->set_floats("i-b-m1", layer.get_i_b_m1());
  layer_object->set_floats("i-b-m2", layer.get_i_b_m2());
  layer_object->set_numbers("i-b-timesteps", layer.get_i_b_timesteps());
  layer_object->set_floats("i-b-decays", layer.get_i_b_decays());

  // Output Gate (o)
  layer_object->set_floats("o-w-values", layer.get_o_w_values());
  layer_object->set_floats("o-w-grads", layer.get_o_w_grads());
  layer_object->set_floats("o-w-velocities", layer.get_o_w_velocities());
  layer_object->set_floats("o-w-m1", layer.get_o_w_m1());
  layer_object->set_floats("o-w-m2", layer.get_o_w_m2());
  layer_object->set_numbers("o-w-timesteps", layer.get_o_w_timesteps());
  layer_object->set_floats("o-w-decays", layer.get_o_w_decays());

  layer_object->set_floats("o-rw-values", layer.get_o_rw_values());
  layer_object->set_floats("o-rw-grads", layer.get_o_rw_grads());
  layer_object->set_floats("o-rw-velocities", layer.get_o_rw_velocities());
  layer_object->set_floats("o-rw-m1", layer.get_o_rw_m1());
  layer_object->set_floats("o-rw-m2", layer.get_o_rw_m2());
  layer_object->set_numbers("o-rw-timesteps", layer.get_o_rw_timesteps());
  layer_object->set_floats("o-rw-decays", layer.get_o_rw_decays());

  layer_object->set_floats("o-b-values", layer.get_o_b_values());
  layer_object->set_floats("o-b-grads", layer.get_o_b_grads());
  layer_object->set_floats("o-b-velocities", layer.get_o_b_velocities());
  layer_object->set_floats("o-b-m1", layer.get_o_b_m1());
  layer_object->set_floats("o-b-m2", layer.get_o_b_m2());
  layer_object->set_numbers("o-b-timesteps", layer.get_o_b_timesteps());
  layer_object->set_floats("o-b-decays", layer.get_o_b_decays());

  layer_object->set_float("momentum", layer.get_momentum());

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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  auto layer_object = new TinyJSON::TJValueObject();
  auto layer_array = new TinyJSON::TJValueArray();
  for (const auto& neuron : layer.get_neurons())
  {
    auto* neuron_object = add_neuron(neuron);
    layer_array->add(neuron_object);
    delete neuron_object;
  }
  layer_object->set_string("layer-name", "fflayer");
  layer_object->set("neurons", layer_array);
  layer_object->set_number("residual-layer-number", layer.get_residual_layer_number());
  layer_object->set_string("optimiser-type", optimiser_type_to_string(layer.get_optimiser_type()).c_str());
  add_activation_helper(layer.get_activation_helper(), *layer_object);
  layer_object->set_number("layer-role", (int)layer.get_layer_role());
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

  layer_object->set_float("momentum", layer.get_momentum());

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

void NeuralNetworkSerializer::add_ffoutputlayer(const FFOutputLayer& layer, TinyJSON::TJValueArray& layers)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  auto layer_object = new TinyJSON::TJValueObject();
  auto layer_array = new TinyJSON::TJValueArray();
  for(const auto& neuron : layer.get_neurons())
  {
    auto* neuron_object = add_neuron(neuron);
    layer_array->add(neuron_object);
    delete neuron_object;
  }
  layer_object->set_string("layer-name", "ffoutputlayer");
  layer_object->set("neurons", layer_array);
  layer_object->set_number("residual-layer-number", layer.get_residual_layer_number());
  layer_object->set_number("layer-role", (int)layer.get_layer_role());

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

  layers.add(layer_object);
  delete layer_array;
  delete layer_object;
}

TinyJSON::TJValueArray* NeuralNetworkSerializer::add_output_layer_details(const std::vector<OutputLayerDetails>& output_layer_details)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  auto output_layer_array = new TinyJSON::TJValueArray();
  for (const auto output_layer_detail : output_layer_details)
  {
    auto output_layer_object = new TinyJSON::TJValueObject();
    output_layer_object->set_number("size", output_layer_detail.get_size());
    output_layer_object->set_string("activation-method", activation::method_to_string(output_layer_detail.get_activation().get_method()).c_str());
    output_layer_object->set_float("activation-alpha", output_layer_detail.get_activation().get_alpha());
    output_layer_object->set_float("activation-temperature", output_layer_detail.get_activation().get_temperature());
    output_layer_object->set_float("activation-inference-temperature", output_layer_detail.get_activation().get_inference_temperature());
    output_layer_object->set_string("error-calculation-type", ErrorCalculation::type_to_string(output_layer_detail.get_output_error_calculation_type()).c_str());
    add_error_evaluation_config(output_layer_object, output_layer_detail.get_error_evaluation_config());
    output_layer_object->set("weight-decay", output_layer_detail.get_weight_decay());
    output_layer_object->set("optimiser-type", optimiser_type_to_string(output_layer_detail.get_optimiser_type()).c_str());
    output_layer_object->set("momentum", output_layer_detail.get_momentum());
    output_layer_array->add(output_layer_object);
    delete output_layer_object;
  }
  return output_layer_array;
}

void NeuralNetworkSerializer::add_error_evaluation_config(TinyJSON::TJValueObject* parent, const EvaluationConfig& config)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  auto error_evaluation_config_object = new TinyJSON::TJValueObject();
  error_evaluation_config_object->set_float("neutral-tolerance", config.neutral_tolerance());
  error_evaluation_config_object->set_float("confidence-threshold", config.confidence_threshold());
  error_evaluation_config_object->set_float("huber-delta", config.huber_delta());
  error_evaluation_config_object->set_float("direction-lambda", config.direction_lambda());
  error_evaluation_config_object->set_boolean("use-direction-penalty", config.use_direction_penalty());
  error_evaluation_config_object->set_float("cross-entropy-lambda", config.cross_entropy_lambda());

  parent->set("error-evaluation-config", error_evaluation_config_object);
  delete error_evaluation_config_object;

}

TinyJSON::TJValueArray* NeuralNetworkSerializer::add_hidden_layers(const std::vector<LayerDetails>& hidden_layers)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  auto hidden_layers_array = new TinyJSON::TJValueArray();
  for (const auto& hl : hidden_layers)
  {
    auto hidden_layer_object = new TinyJSON::TJValueObject();
    hidden_layer_object->set("architecture", Layer::architecture_to_string(hl.get_layer_architecture()).c_str());
    hidden_layer_object->set("size", hl.get_size());
    hidden_layer_object->set("activation-method", activation::method_to_string(hl.get_activation().get_method()).c_str());
    hidden_layer_object->set("activation-alpha", hl.get_activation().get_alpha());
    hidden_layer_object->set_float("activation-temperature", hl.get_activation().get_temperature());
    hidden_layer_object->set_float("activation-inference-temperature", hl.get_activation().get_inference_temperature());
    hidden_layer_object->set("dropout", hl.get_dropout());
    hidden_layer_object->set("weight-decay", hl.get_weight_decay());
    hidden_layer_object->set("optimiser-type", optimiser_type_to_string(hl.get_optimiser_type()).c_str());
    hidden_layer_object->set("momentum", hl.get_momentum());
        
    hidden_layers_array->add(hidden_layer_object);
    delete hidden_layer_object;
  }
  return hidden_layers_array;
}

TinyJSON::TJValueObject* NeuralNetworkSerializer::add_residual_projector(const ResidualProjector* residual_projector)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  auto layers_array = new TinyJSON::TJValueArray();
  const auto& layers = nn.get_layers();
  for(const auto& layer : layers.get_layers())
  {
    add_layer(layer.get(), *layers_array);
  }
  json.set("layers", layers_array);
  delete layers_array;
}

void NeuralNetworkSerializer::add_layer(const Layer* layer, TinyJSON::TJValueArray& layers)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  // FFOutputLayer has to be before FFLayer as it is derived ...
  auto ffoutputlayer = dynamic_cast<const FFOutputLayer*>(layer);
  if (nullptr != ffoutputlayer)
  {
    add_ffoutputlayer(*ffoutputlayer, layers);
    return;
  }

  auto fflayer = dynamic_cast<const FFLayer*>(layer);
  if (nullptr != fflayer)
  {
    add_fflayer(*fflayer, layers);
    return;
  }

  auto elmanrnnlayer = dynamic_cast<const ElmanRNNLayer*>(layer);
  if (nullptr != elmanrnnlayer)
  {
    add_elmanrnnlayer(*elmanrnnlayer, layers);
    return;
  }

  auto grulayer = dynamic_cast<const GRURNNLayer*>(layer);
  if (nullptr != grulayer)
  {
    add_grurnnlayer(*grulayer, layers);
    return;
  }

  auto lstmlayer = dynamic_cast<const LSTMLayer*>(layer);
  if (nullptr != lstmlayer)
  {
    add_lstmlayer(*lstmlayer, layers);
    return;
  }

  auto multioutputlayer = dynamic_cast<const MultiOutputLayer*>(layer);
  if (nullptr != multioutputlayer)
  {
    add_multioutputlayer(*multioutputlayer, layers);
    return;
  }

  Logger::panic("Unknown layer type!");
}

const std::vector<ErrorCalculation::type> NeuralNetworkSerializer::all_error_types()
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  return {
    ErrorCalculation::type::huber_loss,
    ErrorCalculation::type::huber_direction_loss,
    ErrorCalculation::type::mae,
    ErrorCalculation::type::mse,
    ErrorCalculation::type::rmse,
    ErrorCalculation::type::nrmse,
    ErrorCalculation::type::mape,
    ErrorCalculation::type::smape,
    ErrorCalculation::type::wape,
    ErrorCalculation::type::directional_accuracy,
    ErrorCalculation::type::bce_loss,
    ErrorCalculation::type::cross_entropy,
    ErrorCalculation::type::log_cosh,
    ErrorCalculation::type::directional_confidence_score,
    ErrorCalculation::type::prediction_coverage
  };
}

void NeuralNetworkSerializer::add_errors(const NeuralNetwork& nn, TinyJSON::TJValueObject& json)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  const auto& error_types = all_error_types();
  auto tj_errors_array = new TinyJSON::TJValueArray();

  const auto all_metrics = nn.calculate_forecast_metrics_all_layers(error_types);

  for (const auto& metrics : all_metrics)
  {
    auto tj_errors_object = new TinyJSON::TJValueObject();
    for (const auto& metric : metrics)
    {
      tj_errors_object->set_float(ErrorCalculation::type_to_string(metric.error_type()).c_str(), metric.error());
    }
    tj_errors_array->add(tj_errors_object);
    delete tj_errors_object;
  }

  json.set("errors", tj_errors_array);
  delete tj_errors_array;
}

void NeuralNetworkSerializer::add_activation_helper(const layer_activation_helper& lah, TinyJSON::TJValueObject& json)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  auto* ranges_array = new TinyJSON::TJValueArray();
  for (const auto& r : lah.ranges())
  {
    auto* range_object = new TinyJSON::TJValueObject();
    range_object->set_number("start", r.start);
    range_object->set_number("end", r.end);
    range_object->set_string("activation-method", r.activation_method.method_to_string().c_str());
    range_object->set_float("activation-alpha", r.activation_method.get_alpha());
    range_object->set_float("activation-temperature", r.activation_method.get_temperature());
    range_object->set_float("activation-inference-temperature", r.activation_method.get_inference_temperature());
    ranges_array->add(range_object);
    delete range_object;
  }
  json.set("activation-ranges", ranges_array);
  delete ranges_array;
}

void NeuralNetworkSerializer::add_final_learning_rate(const NeuralNetwork& nn, TinyJSON::TJValueObject& json)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  json.set_float("final-learning-rate", nn.get_learning_rate());
}

std::vector<MultiOutputLayerDetails> NeuralNetworkSerializer::get_multi_output_layer_details(const TinyJSON::TJValueObject& options_object)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  const auto* multi_outputs_array = dynamic_cast<const TinyJSON::TJValueArray*>(options_object.try_get_value("multi-output-layers"));
  if (nullptr == multi_outputs_array)
  {
    return {};
  }

  std::vector<MultiOutputLayerDetails> multi_output_layer_details;
  for (const auto& multi_output : *multi_outputs_array)
  {
    const auto* multi_output_object = dynamic_cast<const TinyJSON::TJValueObject*>(&multi_output);
    if (nullptr == multi_output_object)
    {
      Logger::panic("Missing hidden layer(s) details and output layer details for MultiOutputLayerDetails!");
    }

    // Hidden layers
    const auto* hidden_array = dynamic_cast<const TinyJSON::TJValueArray*>(multi_output_object->try_get_value("hidden-layers"));
    if (nullptr == hidden_array)
    {
      Logger::panic("Missing hidden layer(s) details for MultiOutputLayerDetails");
    }

    std::vector<LayerDetails> hidden_layers;
    for (const auto& hl_val : *hidden_array) 
    {
      const auto* phlo = dynamic_cast<const TinyJSON::TJValueObject*>(&hl_val);
      if (!phlo)
      {
        continue;
      }
        
      const auto method_str = phlo->try_get_string("activation-method", false);
      const auto method = activation::string_to_method(method_str == nullptr ? "sigmoid" : method_str);
      const auto alpha = phlo->get<double>("activation-alpha");
      const auto temperature = phlo->get_or<double>("activation-temperature", 1.0);
      const auto inference_temperature = phlo->get_or<double>("activation-inference-temperature", temperature);

      hidden_layers.emplace_back(LayerDetails(
        Layer::architecture_from_string(phlo->try_get_string("architecture", false)),
        phlo->get<unsigned>("size"),
        activation(method, alpha, temperature, inference_temperature),
        phlo->get<double>("dropout"),
        phlo->get<double>("weight-decay"),
        string_to_optimiser_type(phlo->try_get_string("optimiser-type", false)),
        phlo->get<double>("momentum")
      ));
    }

    // Output detail
    const auto* od_obj = dynamic_cast<const TinyJSON::TJValueObject*>(multi_output_object->try_get_value("output-detail"));
    if (nullptr == od_obj)
    {
      Logger::panic("Missing output layer details for MultiOutputLayerDetails");
    }

    const auto method_str = od_obj->try_get_string("activation-method", false);
    const auto method = activation::string_to_method(method_str == nullptr ? "sigmoid" : method_str);
    const auto alpha = (double)od_obj->get_float("activation-alpha");
    const auto temperature = od_obj->get_or<double>("activation-temperature", 1.0);
    const auto inference_temperature = od_obj->get_or<double>("activation-inference-temperature", temperature);

    const auto output_details = OutputLayerDetails(
      od_obj->get<unsigned>("size"),
      activation(method, alpha, temperature, inference_temperature),
      ErrorCalculation::string_to_type(od_obj->get_string("error-calculation-type")),
      get_error_evaluation_config(od_obj),
      od_obj->get<double>("weight-decay"),
      string_to_optimiser_type(od_obj->try_get_string("optimiser-type", false)),
      od_obj->get<double>("momentum"));

    multi_output_layer_details.push_back(MultiOutputLayerDetails(hidden_layers, output_details));
  }
  return multi_output_layer_details;
}

void NeuralNetworkSerializer::add_multioutputlayer(const MultiOutputLayer& layer, TinyJSON::TJValueArray& layers)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  auto layer_object = new TinyJSON::TJValueObject();
  layer_object->set_string("layer-name", "multioutputlayer");
  layer_object->set_number("layer-role", (int)layer.get_layer_role());
  layer_object->set_number("number-input-neurons", layer.get_number_input_neurons());
  layer_object->set_number("number-output-neurons", layer.get_number_output_neurons());
  
  auto branches_array = new TinyJSON::TJValueArray();
  for (const auto& branch : layer.get_branches())
  {
    auto branch_layers_array = new TinyJSON::TJValueArray();
    for (const auto& branch_layer : branch.layers)
    {
      add_layer(branch_layer.get(), *branch_layers_array);
    }
    branches_array->add(branch_layers_array);
    delete branch_layers_array;
  }
  layer_object->set("branches", branches_array);
  delete branches_array;

  add_activation_helper(layer.get_activation_helper(), *layer_object);

  layers.add(layer_object);
  delete layer_object;
}

std::unique_ptr<Layer> NeuralNetworkSerializer::create_multioutputlayer(
  unsigned layer_index, 
  const TinyJSON::TJValueObject& layer_object, 
  int number_of_threads,
  const std::vector<MultiOutputLayerDetails>& multi_output_layer_details
)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  auto number_input_neurons = layer_object.get<unsigned>("number-input-neurons");
  auto number_output_neurons = layer_object.get<unsigned>("number-output-neurons");
  // has-bias would ideally be in the layer_object, defaulting to true
  bool has_bias = true; 

  auto layer = std::make_unique<MultiOutputLayer>(
    layer_index,
    number_input_neurons,
    number_output_neurons,
    multi_output_layer_details,
    number_of_threads,
    has_bias
  );

  auto* branches_array = dynamic_cast<const TinyJSON::TJValueArray*>(layer_object.try_get_value("branches"));
  if (branches_array != nullptr)
  {
    auto& branches = layer->get_mutable_branches();
    for (size_t b_idx = 0; b_idx < branches_array->get_number_of_items() && b_idx < branches.size(); ++b_idx)
    {
      auto* branch_layers_array = dynamic_cast<const TinyJSON::TJValueArray*>(branches_array->at(static_cast<unsigned>(b_idx)));
      if (branch_layers_array != nullptr)
      {
        auto& branch = branches[b_idx];
        for (size_t l_idx = 0; l_idx < branch_layers_array->get_number_of_items() && l_idx < branch.layers.size(); ++l_idx)
        {
          auto* b_layer_obj = dynamic_cast<const TinyJSON::TJValueObject*>(branch_layers_array->at(static_cast<unsigned>(l_idx)));
          if (b_layer_obj != nullptr)
          {
            load_weights(*branch.layers[l_idx], *b_layer_obj);
          }
        }
      }
    }
  }

  return layer;
}

void NeuralNetworkSerializer::load_weights(Layer& layer, const TinyJSON::TJValueObject& layer_object)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkSerializer");
  
  // Basic weights and biases
  if (layer_object.has_key("w-values"))
  {
    layer.set_w_values(layer_object.get<std::vector<double>>("w-values"));
  }
  if (layer_object.has_key("w-grads"))
  {
    layer.set_w_grads(layer_object.get<std::vector<double>>("w-grads"));
  }
  if (layer_object.has_key("w-velocities"))
  {
    layer.set_w_velocities(layer_object.get<std::vector<double>>("w-velocities"));
  }
  if (layer_object.has_key("w-m1"))
  {
    layer.set_w_m1(layer_object.get<std::vector<double>>("w-m1"));
  }
  if (layer_object.has_key("w-m2"))
  {
    layer.set_w_m2(layer_object.get<std::vector<double>>("w-m2"));
  }
  if (layer_object.has_key("w-timesteps"))
  {
    layer.set_w_timesteps(layer_object.get<std::vector<long long>>("w-timesteps"));
  }
  if (layer_object.has_key("w-decays"))
  {
    layer.set_w_decays(layer_object.get<std::vector<double>>("w-decays"));
  }

  if (layer_object.has_key("b-values"))
  {
    layer.set_b_values(layer_object.get<std::vector<double>>("b-values"));
  }
  if (layer_object.has_key("b-grads"))
  {
    layer.set_b_grads(layer_object.get<std::vector<double>>("b-grads"));
  }
  if (layer_object.has_key("b-velocities"))
  {
    layer.set_b_velocities(layer_object.get<std::vector<double>>("b-velocities"));
  }
  if (layer_object.has_key("b-m1"))
  {
    layer.set_b_m1(layer_object.get<std::vector<double>>("b-m1"));
  }
  if (layer_object.has_key("b-m2"))
  {
    layer.set_b_m2(layer_object.get<std::vector<double>>("b-m2"));
  }
  if (layer_object.has_key("b-timesteps"))
  {
    layer.set_b_timesteps(layer_object.get<std::vector<long long>>("b-timesteps"));
  }
  if (layer_object.has_key("b-decays"))
  {
    layer.set_b_decays(layer_object.get<std::vector<double>>("b-decays"));
  }

  // Recurrent weights (if applicable)
  auto* rnn_layer = dynamic_cast<ElmanRNNLayer*>(&layer);
  auto* gru_layer = dynamic_cast<GRURNNLayer*>(&layer);
  auto* lstm_layer = dynamic_cast<LSTMLayer*>(&layer);

  if (rnn_layer || gru_layer || lstm_layer)
  {
    if (layer_object.has_key("rw-values"))
    {
      layer.set_rw_values(layer_object.get<std::vector<double>>("rw-values"));
    }
    if (layer_object.has_key("rw-grads"))
    {
      layer.set_rw_grads(layer_object.get<std::vector<double>>("rw-grads"));
    }
    if (layer_object.has_key("rw-velocities"))
    {
      layer.set_rw_velocities(layer_object.get<std::vector<double>>("rw-velocities"));
    }
    if (layer_object.has_key("rw-m1"))
    {
      layer.set_rw_m1(layer_object.get<std::vector<double>>("rw-m1"));
    }
    if (layer_object.has_key("rw-m2"))
    {
      layer.set_rw_m2(layer_object.get<std::vector<double>>("rw-m2"));
    }
    if (layer_object.has_key("rw-timesteps"))
    {
      layer.set_rw_timesteps(layer_object.get<std::vector<long long>>("rw-timesteps"));
    }
    if (layer_object.has_key("rw-decays"))
    {
      layer.set_rw_decays(layer_object.get<std::vector<double>>("rw-decays"));
    }
  }

  // GRU specific gates
  if (gru_layer)
  {
    if (layer_object.has_key("z-w-values"))
    {
      gru_layer->set_z_w_values(layer_object.get<std::vector<double>>("z-w-values"));
    }
    if (layer_object.has_key("z-w-grads"))
    {
      gru_layer->set_z_w_grads(layer_object.get<std::vector<double>>("z-w-grads"));
    }
    if (layer_object.has_key("z-w-velocities"))
    {
      gru_layer->set_z_w_velocities(layer_object.get<std::vector<double>>("z-w-velocities"));
    }
    if (layer_object.has_key("z-w-m1"))
    {
      gru_layer->set_z_w_m1(layer_object.get<std::vector<double>>("z-w-m1"));
    }
    if (layer_object.has_key("z-w-m2"))
    {
      gru_layer->set_z_w_m2(layer_object.get<std::vector<double>>("z-w-m2"));
    }
    if (layer_object.has_key("z-w-timesteps"))
    {
      gru_layer->set_z_w_timesteps(layer_object.get<std::vector<long long>>("z-w-timesteps"));
    }
    if (layer_object.has_key("z-w-decays"))
    {
      gru_layer->set_z_w_decays(layer_object.get<std::vector<double>>("z-w-decays"));
    }

    if (layer_object.has_key("z-rw-values"))
    {
      gru_layer->set_z_rw_values(layer_object.get<std::vector<double>>("z-rw-values"));
    }
    if (layer_object.has_key("z-rw-grads"))
    {
      gru_layer->set_z_rw_grads(layer_object.get<std::vector<double>>("z-rw-grads"));
    }
    if (layer_object.has_key("z-rw-velocities"))
    {
      gru_layer->set_z_rw_velocities(layer_object.get<std::vector<double>>("z-rw-velocities"));
    }
    if (layer_object.has_key("z-rw-m1"))
    {
      gru_layer->set_z_rw_m1(layer_object.get<std::vector<double>>("z-rw-m1"));
    }
    if (layer_object.has_key("z-rw-m2"))
    {
      gru_layer->set_z_rw_m2(layer_object.get<std::vector<double>>("z-rw-m2"));
    }
    if (layer_object.has_key("z-rw-timesteps"))
    {
      gru_layer->set_z_rw_timesteps(layer_object.get<std::vector<long long>>("z-rw-timesteps"));
    }
    if (layer_object.has_key("z-rw-decays"))
    {
      gru_layer->set_z_rw_decays(layer_object.get<std::vector<double>>("z-rw-decays"));
    }

    if (layer_object.has_key("z-b-values"))
    {
      gru_layer->set_z_b_values(layer_object.get<std::vector<double>>("z-b-values"));
    }
    if (layer_object.has_key("z-b-grads"))
    {
      gru_layer->set_z_b_grads(layer_object.get<std::vector<double>>("z-b-grads"));
    }
    if (layer_object.has_key("z-b-velocities"))
    {
      gru_layer->set_z_b_velocities(layer_object.get<std::vector<double>>("z-b-velocities"));
    }
    if (layer_object.has_key("z-b-m1"))
    {
      gru_layer->set_z_b_m1(layer_object.get<std::vector<double>>("z-b-m1"));
    }
    if (layer_object.has_key("z-b-m2"))
    {
      gru_layer->set_z_b_m2(layer_object.get<std::vector<double>>("z-b-m2"));
    }
    if (layer_object.has_key("z-b-timesteps"))
    {
      gru_layer->set_z_b_timesteps(layer_object.get<std::vector<long long>>("z-b-timesteps"));
    }
    if (layer_object.has_key("z-b-decays"))
    {
      gru_layer->set_z_b_decays(layer_object.get<std::vector<double>>("z-b-decays"));
    }

    if (layer_object.has_key("r-w-values"))
    {
      gru_layer->set_r_w_values(layer_object.get<std::vector<double>>("r-w-values"));
    }
    if (layer_object.has_key("r-w-grads"))
    {
      gru_layer->set_r_w_grads(layer_object.get<std::vector<double>>("r-w-grads"));
    }
    if (layer_object.has_key("r-w-velocities"))
    {
      gru_layer->set_r_w_velocities(layer_object.get<std::vector<double>>("r-w-velocities"));
    }
    if (layer_object.has_key("r-w-m1"))
    {
      gru_layer->set_r_w_m1(layer_object.get<std::vector<double>>("r-w-m1"));
    }
    if (layer_object.has_key("r-w-m2"))
    {
      gru_layer->set_r_w_m2(layer_object.get<std::vector<double>>("r-w-m2"));
    }
    if (layer_object.has_key("r-w-timesteps"))
    {
      gru_layer->set_r_w_timesteps(layer_object.get<std::vector<long long>>("r-w-timesteps"));
    }
    if (layer_object.has_key("r-w-decays"))
    {
      gru_layer->set_r_w_decays(layer_object.get<std::vector<double>>("r-w-decays"));
    }

    if (layer_object.has_key("r-rw-values"))
    {
      gru_layer->set_r_rw_values(layer_object.get<std::vector<double>>("r-rw-values"));
    }
    if (layer_object.has_key("r-rw-grads"))
    {
      gru_layer->set_r_rw_grads(layer_object.get<std::vector<double>>("r-rw-grads"));
    }
    if (layer_object.has_key("r-rw-velocities"))
    {
      gru_layer->set_r_rw_velocities(layer_object.get<std::vector<double>>("r-rw-velocities"));
    }
    if (layer_object.has_key("r-rw-m1"))
    {
      gru_layer->set_r_rw_m1(layer_object.get<std::vector<double>>("r-rw-m1"));
    }
    if (layer_object.has_key("r-rw-m2"))
    {
      gru_layer->set_r_rw_m2(layer_object.get<std::vector<double>>("r-rw-m2"));
    }
    if (layer_object.has_key("r-rw-timesteps"))
    {
      gru_layer->set_r_rw_timesteps(layer_object.get<std::vector<long long>>("r-rw-timesteps"));
    }
    if (layer_object.has_key("r-rw-decays"))
    {
      gru_layer->set_r_rw_decays(layer_object.get<std::vector<double>>("r-rw-decays"));
    }

    if (layer_object.has_key("r-b-values"))
    {
      gru_layer->set_r_b_values(layer_object.get<std::vector<double>>("r-b-values"));
    }
    if (layer_object.has_key("r-b-grads"))
    {
      gru_layer->set_r_b_grads(layer_object.get<std::vector<double>>("r-b-grads"));
    }
    if (layer_object.has_key("r-b-velocities"))
    {
      gru_layer->set_r_b_velocities(layer_object.get<std::vector<double>>("r-b-velocities"));
    }
    if (layer_object.has_key("r-b-m1"))
    {
      gru_layer->set_r_b_m1(layer_object.get<std::vector<double>>("r-b-m1"));
    }
    if (layer_object.has_key("r-b-m2"))
    {
      gru_layer->set_r_b_m2(layer_object.get<std::vector<double>>("r-b-m2"));
    }
    if (layer_object.has_key("r-b-timesteps"))
    {
      gru_layer->set_r_b_timesteps(layer_object.get<std::vector<long long>>("r-b-timesteps"));
    }
    if (layer_object.has_key("r-b-decays"))
    {
      gru_layer->set_r_b_decays(layer_object.get<std::vector<double>>("r-b-decays"));
    }
  }

  // LSTM specific gates
  if (lstm_layer)
  {
    // Forget Gate
    if (layer_object.has_key("f-w-values")) lstm_layer->set_f_w_values(layer_object.get<std::vector<double>>("f-w-values"));
    if (layer_object.has_key("f-w-grads")) lstm_layer->set_f_w_grads(layer_object.get<std::vector<double>>("f-w-grads"));
    if (layer_object.has_key("f-w-velocities")) lstm_layer->set_f_w_velocities(layer_object.get<std::vector<double>>("f-w-velocities"));
    if (layer_object.has_key("f-w-m1")) lstm_layer->set_f_w_m1(layer_object.get<std::vector<double>>("f-w-m1"));
    if (layer_object.has_key("f-w-m2")) lstm_layer->set_f_w_m2(layer_object.get<std::vector<double>>("f-w-m2"));
    if (layer_object.has_key("f-w-timesteps")) lstm_layer->set_f_w_timesteps(layer_object.get<std::vector<long long>>("f-w-timesteps"));
    if (layer_object.has_key("f-w-decays")) lstm_layer->set_f_w_decays(layer_object.get<std::vector<double>>("f-w-decays"));

    if (layer_object.has_key("f-rw-values")) lstm_layer->set_f_rw_values(layer_object.get<std::vector<double>>("f-rw-values"));
    if (layer_object.has_key("f-rw-grads")) lstm_layer->set_f_rw_grads(layer_object.get<std::vector<double>>("f-rw-grads"));
    if (layer_object.has_key("f-rw-velocities")) lstm_layer->set_f_rw_velocities(layer_object.get<std::vector<double>>("f-rw-velocities"));
    if (layer_object.has_key("f-rw-m1")) lstm_layer->set_f_rw_m1(layer_object.get<std::vector<double>>("f-rw-m1"));
    if (layer_object.has_key("f-rw-m2")) lstm_layer->set_f_rw_m2(layer_object.get<std::vector<double>>("f-rw-m2"));
    if (layer_object.has_key("f-rw-timesteps")) lstm_layer->set_f_rw_timesteps(layer_object.get<std::vector<long long>>("f-rw-timesteps"));
    if (layer_object.has_key("f-rw-decays")) lstm_layer->set_f_rw_decays(layer_object.get<std::vector<double>>("f-rw-decays"));

    if (layer_object.has_key("f-b-values")) lstm_layer->set_f_b_values(layer_object.get<std::vector<double>>("f-b-values"));
    if (layer_object.has_key("f-b-grads")) lstm_layer->set_f_b_grads(layer_object.get<std::vector<double>>("f-b-grads"));
    if (layer_object.has_key("f-b-velocities")) lstm_layer->set_f_b_velocities(layer_object.get<std::vector<double>>("f-b-velocities"));
    if (layer_object.has_key("f-b-m1")) lstm_layer->set_f_b_m1(layer_object.get<std::vector<double>>("f-b-m1"));
    if (layer_object.has_key("f-b-m2")) lstm_layer->set_f_b_m2(layer_object.get<std::vector<double>>("f-b-m2"));
    if (layer_object.has_key("f-b-timesteps")) lstm_layer->set_f_b_timesteps(layer_object.get<std::vector<long long>>("f-b-timesteps"));
    if (layer_object.has_key("f-b-decays")) lstm_layer->set_f_b_decays(layer_object.get<std::vector<double>>("f-b-decays"));

    // Input Gate
    if (layer_object.has_key("i-w-values")) lstm_layer->set_i_w_values(layer_object.get<std::vector<double>>("i-w-values"));
    if (layer_object.has_key("i-w-grads")) lstm_layer->set_i_w_grads(layer_object.get<std::vector<double>>("i-w-grads"));
    if (layer_object.has_key("i-w-velocities")) lstm_layer->set_i_w_velocities(layer_object.get<std::vector<double>>("i-w-velocities"));
    if (layer_object.has_key("i-w-m1")) lstm_layer->set_i_w_m1(layer_object.get<std::vector<double>>("i-w-m1"));
    if (layer_object.has_key("i-w-m2")) lstm_layer->set_i_w_m2(layer_object.get<std::vector<double>>("i-w-m2"));
    if (layer_object.has_key("i-w-timesteps")) lstm_layer->set_i_w_timesteps(layer_object.get<std::vector<long long>>("i-w-timesteps"));
    if (layer_object.has_key("i-w-decays")) lstm_layer->set_i_w_decays(layer_object.get<std::vector<double>>("i-w-decays"));

    if (layer_object.has_key("i-rw-values")) lstm_layer->set_i_rw_values(layer_object.get<std::vector<double>>("i-rw-values"));
    if (layer_object.has_key("i-rw-grads")) lstm_layer->set_i_rw_grads(layer_object.get<std::vector<double>>("i-rw-grads"));
    if (layer_object.has_key("i-rw-velocities")) lstm_layer->set_i_rw_velocities(layer_object.get<std::vector<double>>("i-rw-velocities"));
    if (layer_object.has_key("i-rw-m1")) lstm_layer->set_i_rw_m1(layer_object.get<std::vector<double>>("i-rw-m1"));
    if (layer_object.has_key("i-rw-m2")) lstm_layer->set_i_rw_m2(layer_object.get<std::vector<double>>("i-rw-m2"));
    if (layer_object.has_key("i-rw-timesteps")) lstm_layer->set_i_rw_timesteps(layer_object.get<std::vector<long long>>("i-rw-timesteps"));
    if (layer_object.has_key("i-rw-decays")) lstm_layer->set_i_rw_decays(layer_object.get<std::vector<double>>("i-rw-decays"));

    if (layer_object.has_key("i-b-values")) lstm_layer->set_i_b_values(layer_object.get<std::vector<double>>("i-b-values"));
    if (layer_object.has_key("i-b-grads")) lstm_layer->set_i_b_grads(layer_object.get<std::vector<double>>("i-b-grads"));
    if (layer_object.has_key("i-b-velocities")) lstm_layer->set_i_b_velocities(layer_object.get<std::vector<double>>("i-b-velocities"));
    if (layer_object.has_key("i-b-m1")) lstm_layer->set_i_b_m1(layer_object.get<std::vector<double>>("i-b-m1"));
    if (layer_object.has_key("i-b-m2")) lstm_layer->set_i_b_m2(layer_object.get<std::vector<double>>("i-b-m2"));
    if (layer_object.has_key("i-b-timesteps")) lstm_layer->set_i_b_timesteps(layer_object.get<std::vector<long long>>("i-b-timesteps"));
    if (layer_object.has_key("i-b-decays")) lstm_layer->set_i_b_decays(layer_object.get<std::vector<double>>("i-b-decays"));

    // Output Gate
    if (layer_object.has_key("o-w-values")) lstm_layer->set_o_w_values(layer_object.get<std::vector<double>>("o-w-values"));
    if (layer_object.has_key("o-w-grads")) lstm_layer->set_o_w_grads(layer_object.get<std::vector<double>>("o-w-grads"));
    if (layer_object.has_key("o-w-velocities")) lstm_layer->set_o_w_velocities(layer_object.get<std::vector<double>>("o-w-velocities"));
    if (layer_object.has_key("o-w-m1")) lstm_layer->set_o_w_m1(layer_object.get<std::vector<double>>("o-w-m1"));
    if (layer_object.has_key("o-w-m2")) lstm_layer->set_o_w_m2(layer_object.get<std::vector<double>>("o-w-m2"));
    if (layer_object.has_key("o-w-timesteps")) lstm_layer->set_o_w_timesteps(layer_object.get<std::vector<long long>>("o-w-timesteps"));
    if (layer_object.has_key("o-w-decays")) lstm_layer->set_o_w_decays(layer_object.get<std::vector<double>>("o-w-decays"));

    if (layer_object.has_key("o-rw-values")) lstm_layer->set_o_rw_values(layer_object.get<std::vector<double>>("o-rw-values"));
    if (layer_object.has_key("o-rw-grads")) lstm_layer->set_o_rw_grads(layer_object.get<std::vector<double>>("o-rw-grads"));
    if (layer_object.has_key("o-rw-velocities")) lstm_layer->set_o_rw_velocities(layer_object.get<std::vector<double>>("o-rw-velocities"));
    if (layer_object.has_key("o-rw-m1")) lstm_layer->set_o_rw_m1(layer_object.get<std::vector<double>>("o-rw-m1"));
    if (layer_object.has_key("o-rw-m2")) lstm_layer->set_o_rw_m2(layer_object.get<std::vector<double>>("o-rw-m2"));
    if (layer_object.has_key("o-rw-timesteps")) lstm_layer->set_o_rw_timesteps(layer_object.get<std::vector<long long>>("o-rw-timesteps"));
    if (layer_object.has_key("o-rw-decays")) lstm_layer->set_o_rw_decays(layer_object.get<std::vector<double>>("o-rw-decays"));

    if (layer_object.has_key("o-b-values")) lstm_layer->set_o_b_values(layer_object.get<std::vector<double>>("o-b-values"));
    if (layer_object.has_key("o-b-grads")) lstm_layer->set_o_b_grads(layer_object.get<std::vector<double>>("o-b-grads"));
    if (layer_object.has_key("o-b-velocities")) lstm_layer->set_o_b_velocities(layer_object.get<std::vector<double>>("o-b-velocities"));
    if (layer_object.has_key("o-b-m1")) lstm_layer->set_o_b_m1(layer_object.get<std::vector<double>>("o-b-m1"));
    if (layer_object.has_key("o-b-m2")) lstm_layer->set_o_b_m2(layer_object.get<std::vector<double>>("o-b-m2"));
    if (layer_object.has_key("o-b-timesteps")) lstm_layer->set_o_b_timesteps(layer_object.get<std::vector<long long>>("o-b-timesteps"));
    if (layer_object.has_key("o-b-decays")) lstm_layer->set_o_b_decays(layer_object.get<std::vector<double>>("o-b-decays"));
  }

  // Residual projector
  if (layer_object.has_key("residual-projector"))
  {
    auto projector = get_residual_projector(layer_object);
    layer.set_residual_projector(projector);
  }
}
