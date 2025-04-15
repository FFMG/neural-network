#include "neuralnetworkserializer.h"

NeuralNetworkSerializer::NeuralNetworkSerializer()
{

}

NeuralNetwork* NeuralNetworkSerializer::load(const std::string& path)
{
  //  any errors will throw
  auto* tj = TinyJSON::TJ::parse_file(path.c_str());

  // cleanup
  delete tj;

  // return the saved file.s
  return nullptr;
}

void NeuralNetworkSerializer::save(const NeuralNetwork& nn, const std::string& path)
{
  // create the object.
  auto tj = new TinyJSON::TJValueObject();
  add_topology(nn, *tj);
  add_layers(nn, *tj);

  // save it.
  TinyJSON::TJ::write_file(path.c_str(), *tj);
  
  // cleanup
  delete tj;
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
  neuron.set("weight", weights_array);
  delete weights_array;
}

void NeuralNetworkSerializer::add_neuron(const Neuron& neuron, TinyJSON::TJValueArray& layer)
{
  auto neuron_object = new TinyJSON::TJValueObject();
  neuron_object->set_number("index", neuron.get_index());
  neuron_object->set_float("learning_rate", neuron.get_learning_rate());
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