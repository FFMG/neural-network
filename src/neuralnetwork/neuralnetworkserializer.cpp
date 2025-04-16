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
  std::vector<Neuron::Layer*> players;
  std::vector<Neuron::Layer> layers;
  for(auto layer_number = 0; ;++layer_number)
  {
    auto layer = get_layer(*tj, layer_number, activation_method);
    if(layer == nullptr)
    {
      break;
    }
    layers.push_back(*layer);
    players.push_back(layer);
  }

  // validate that the topology matches.

  // create the NN
  auto nn = new NeuralNetwork(layers, activation_method);
  for(auto layer : players)
  {
    delete layer;
  }
  layers.clear();

  // cleanup
  delete tj;
  return nn;
}

void NeuralNetworkSerializer::save(const NeuralNetwork& nn, const std::string& path)
{
  // create the object.
  auto tj = new TinyJSON::TJValueObject();
  add_topology(nn, *tj);
  add_activation_method(nn, *tj);
  add_layers(nn, *tj);

  // save it.
  TinyJSON::TJ::write_file(path.c_str(), *tj);
  
  // cleanup
  delete tj;
}

Neuron::Layer* NeuralNetworkSerializer::get_layer(
  const TinyJSON::TJValue& json, 
  unsigned layer_number,
  const activation::method& activation_method)
{
  auto object = dynamic_cast<const TinyJSON::TJValueObject*>(&json);
  if(nullptr == object)
  {
    return nullptr;
  }
  auto array = dynamic_cast<const TinyJSON::TJValueArray*>(object->try_get_value("layers"));
  if(nullptr == array)
  {
    return nullptr;
  }
  if(layer_number >= array->get_number_of_items())
  {
    return nullptr;    
  }

  auto layer_array = dynamic_cast<const TinyJSON::TJValueArray*>(array->at(layer_number));
  if(nullptr == layer_array)
  {
    return nullptr;    
  }

  auto layer = new Neuron::Layer();
  for( auto i = 0; i < layer_array->get_number_of_items(); ++i)
  {
    auto neuron_object = dynamic_cast<const TinyJSON::TJValueObject*>(layer_array->at(i));
    if(nullptr == neuron_object)
    {
      delete layer;
      return nullptr;
    }
    auto index_object = dynamic_cast<const TinyJSON::TJValueNumber*>(neuron_object->try_get_value("index"));
    if(nullptr == index_object)
    {
      delete layer;
      return nullptr;
    }
    auto learning_rate_object = dynamic_cast<const TinyJSON::TJValueNumber*>(neuron_object->try_get_value("learning_rate"));
    if(nullptr == learning_rate_object)
    {
      delete layer;
      return nullptr;
    }
    auto gradient_object = dynamic_cast<const TinyJSON::TJValueNumber*>(neuron_object->try_get_value("gradient"));
    if(nullptr == gradient_object)
    {
      delete layer;
      return nullptr;
    }
    auto output_value_object = dynamic_cast<const TinyJSON::TJValueNumber*>(neuron_object->try_get_value("output_value"));
    if(nullptr == output_value_object)
    {
      delete layer;
      return nullptr;
    }

    auto index = index_object->get_number();
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
      weights
    );
    layer->push_back(neuron);
  }
  return layer;
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
  for(auto i = 0; i < weights_array->get_number_of_items(); ++i)
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
  for( auto i = 0; i < array->get_number_of_items(); ++i)
  {
    auto number = dynamic_cast<const TinyJSON::TJValueNumberInt*>(array->at(i));
    if(nullptr == number)
    {
      return {};
    }
    topology.push_back(number->get_number());
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

void NeuralNetworkSerializer::add_layer(const Neuron::Layer& layer, TinyJSON::TJValueArray& layers)
{
  auto layer_array = new TinyJSON::TJValueArray();
  for(auto neuron : layer)
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