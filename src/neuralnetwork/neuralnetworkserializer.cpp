#include <chrono>

#include "neuralnetworkserializer.h"

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
    return nullptr;
  }

  auto topology = get_topology(*tj);
  auto activation_method = get_activation_method(*tj);

  // get the weights...
  std::vector<std::vector<Neuron>> array_of_neurons;
  for(auto layer_number = 0; ;++layer_number)
  {
    auto neurons = get_neurons(*tj, layer_number, activation_method);
    if(neurons.size() == 0)
    {
      break;
    }
    array_of_neurons.push_back(neurons);
  }

  auto error = get_error(*tj);
  auto learning_rate = get_learning_rate(*tj);

  // create the layer and validate that the topology matches.
  auto layers = create_layers(array_of_neurons);
  if(layers.size() == 0 )
  {
    delete tj;
    return nullptr;
  }

  // create the NN
  auto nn = new NeuralNetwork(layers, activation_method, learning_rate, error);

  // cleanup
  delete tj;
  return nn;
}

std::vector<Layer> NeuralNetworkSerializer::create_layers(std::vector<std::vector<Neuron>> array_of_neurons)
{
  std::vector<Layer> layers = {};
  auto number_of_layers = array_of_neurons.size();
  if(number_of_layers <= 2)
  {
    std::cerr << "The number of layers must be at least 2, (input+output)";
    return {};
  }

  // add the input layer
  auto input_neurons = array_of_neurons.front();
  layers.push_back(Layer::create_input_layer(input_neurons));
  
  // create the hidden layers.
  for(size_t i = 1; i < number_of_layers -1; ++i)
  {
    const auto num_neurons_in_previous_layer = static_cast<unsigned>(array_of_neurons[i - 1].size());
    const auto& this_neurons = array_of_neurons[i];
    layers.push_back(Layer::create_hidden_layer(this_neurons, num_neurons_in_previous_layer));
  }

  // finally, the output layer.
  auto output_neurons = array_of_neurons.back();
  const auto num_neurons_in_previous_layer = static_cast<unsigned>(array_of_neurons[array_of_neurons.size()-2].size());
  layers.push_back(Layer::create_output_layer(output_neurons, num_neurons_in_previous_layer));
  
  return layers;
}

void NeuralNetworkSerializer::save(const NeuralNetwork& nn, const std::string& path)
{
  // create the object.
  auto tj = new TinyJSON::TJValueObject();
  add_basic(nn, *tj);
  add_topology(nn, *tj);
  add_activation_method(nn, *tj);
  add_error(nn, *tj);
  add_layers(nn, *tj);
  
  // save it.
  TinyJSON::TJ::write_file(path.c_str(), *tj);
  
  // cleanup
  delete tj;
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
    return LEARNING_RATE;
  }
  return object->get_float("learning-rate", true, false);
}

std::vector<Neuron> NeuralNetworkSerializer::get_neurons(const TinyJSON::TJValue& json, unsigned layer_number,const activation::method& activation_method)
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
  for( unsigned i = 0; i < layer_array->get_number_of_items(); ++i)
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
    auto learning_rate_object = dynamic_cast<const TinyJSON::TJValueNumber*>(neuron_object->try_get_value("learning_rate"));
    if(nullptr == learning_rate_object)
    {
      return {};
    }
    auto gradient_object = dynamic_cast<const TinyJSON::TJValueNumber*>(neuron_object->try_get_value("gradient"));
    if(nullptr == gradient_object)
    {
      return {};
    }
    auto output_value_object = dynamic_cast<const TinyJSON::TJValueNumber*>(neuron_object->try_get_value("output_value"));
    if(nullptr == output_value_object)
    {
      return {};
    }

    auto index = static_cast<unsigned>(index_object->get_number());
    auto output_value = output_value_object->get_float();
    auto gradient = gradient_object->get_float();
    auto learning_rate = learning_rate_object->get_float();

    // then the weights
    // the output layer can have zero weights
    auto weights = get_weights(*neuron_object);
    
    auto neuron = Neuron(
      index,
      output_value,
      gradient,
      activation_method,
      weights,
      learning_rate
    );
    neurons.push_back(neuron);
  }
  return neurons;
}

std::vector<std::array<double,2>> NeuralNetworkSerializer::get_weights(const TinyJSON::TJValueObject& neuron)
{
  // the array of weight
  auto weights_array = dynamic_cast<const TinyJSON::TJValueArray*>(neuron.try_get_value("weights"));
  if(nullptr == weights_array)
  {
    return {};
  }

  std::vector<std::array<double,2>> weights;
  for(unsigned i = 0; i < weights_array->get_number_of_items(); ++i)
  {
    auto inner_weights_array = dynamic_cast<const TinyJSON::TJValueArray*>(weights_array->at(i));
    if( nullptr == inner_weights_array || 2 != inner_weights_array->get_number_of_items())
    {
      return {};
    }
    auto weight1 = dynamic_cast<const TinyJSON::TJValueNumber*>(inner_weights_array->at(0));
    auto weight2 = dynamic_cast<const TinyJSON::TJValueNumber*>(inner_weights_array->at(1));
    if( nullptr == weight1 || nullptr == weight2)
    {
      return {};
    }
    weights.push_back({(double)weight1->get_float(), (double)weight2->get_float()});
  }
  return weights;
}

std::vector<unsigned> NeuralNetworkSerializer::get_topology(const TinyJSON::TJValue& json )
{
  auto object = dynamic_cast<const TinyJSON::TJValueObject*>(&json);
  if(nullptr == object)
  {
    return {};
  }
  auto array = dynamic_cast<const TinyJSON::TJValueArray*>(object->try_get_value("topology"));
  if(nullptr == array)
  {
    return {};
  }

  std::vector<unsigned> topology;
  for( unsigned i = 0; i < array->get_number_of_items(); ++i)
  {
    auto number = dynamic_cast<const TinyJSON::TJValueNumberInt*>(array->at(i));
    if(nullptr == number)
    {
      return {};
    }
    topology.push_back(static_cast<unsigned>(number->get_number()));
  }
  return topology;
}

activation::method NeuralNetworkSerializer::get_activation_method(const TinyJSON::TJValue& json )
{
  auto object = dynamic_cast<const TinyJSON::TJValueObject*>(&json);
  if(nullptr == object)
  {
    return activation::method::sigmoid_activation;
  }
  auto number = dynamic_cast<const TinyJSON::TJValueNumberInt*>(object->try_get_value("activation_method"));
  if(nullptr == number)
  {
    return activation::method::sigmoid_activation;
  }
  return static_cast<activation::method>(number->get_number());
}

void NeuralNetworkSerializer::add_weights(const std::vector<std::array<double,2>>& weights, TinyJSON::TJValueObject& neuron)
{
  auto weights_array = new TinyJSON::TJValueArray();
  for( auto weight : weights)
  {
    auto weight_array = new TinyJSON::TJValueArray();
    weight_array->add_float(weight[0]);
    weight_array->add_float(weight[1]);
    weights_array->add(weight_array);
    delete weight_array;
  }
  neuron.set("weights", weights_array);
  delete weights_array;
}

void NeuralNetworkSerializer::add_basic(const NeuralNetwork& nn, TinyJSON::TJValueObject& json)
{
  auto now = std::chrono::system_clock::now();
  auto now_seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
  long long current_timestamp = now_seconds.time_since_epoch().count();

  json.set_number("created", current_timestamp);
  json.set_float("learning-rate", nn.get_learning_rate());
}

void NeuralNetworkSerializer::add_neuron(const Neuron& neuron, TinyJSON::TJValueArray& layer)
{
  auto neuron_object = new TinyJSON::TJValueObject();
  neuron_object->set_number("index", neuron.get_index());
  neuron_object->set_float("learning_rate", neuron.get_learning_rate());
  neuron_object->set_float("output_value", neuron.get_output_value());
  neuron_object->set_float("gradient", neuron.get_gradient());
  add_weights(neuron.get_weights(), *neuron_object);
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
  for(auto layer : nn.get_layers())
  {
    add_layer(layer, *layers_array);
  }
  json.set("layers", layers_array);
  delete layers_array;
}

void NeuralNetworkSerializer::add_error(const NeuralNetwork& nn, TinyJSON::TJValueObject& json)
{
  auto error = new TinyJSON::TJValueNumberFloat(nn.get_error());
  json.set("error", error);
  delete error;
}

void NeuralNetworkSerializer::add_activation_method(const NeuralNetwork& nn, TinyJSON::TJValueObject& json)
{
  auto activation_method = new TinyJSON::TJValueNumberInt( static_cast<unsigned>(nn.get_activation_method()));
  json.set("activation_method", activation_method);
  delete activation_method;
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