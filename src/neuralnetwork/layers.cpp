#include "elmanrnnlayer.h"
#include "fflayer.h"
#include "ffoutputlayer.h"
#include "grurnnlayer.h"
#include "layers.h"

Layers::Layers(const NeuralNetworkOptions& options) noexcept :
  _update_weights_pool(nullptr)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  _update_weights_pool = new TaskQueuePool<void>(options.number_of_threads());

  const auto& topology = options.topology();

  const auto& hidden_layers = options.hidden_layers();
#if VALIDATE_DATA == 1
  if(hidden_layers.size() != topology.size() - 2)
  {
    Logger::panic("The topology size does not match the layer details size!");
  }
#endif

  const auto& number_of_layers = topology.size();
  _layers.reserve(number_of_layers);

  // add the input layer
  const auto& number_of_threads = options.number_of_threads();
  auto layer = create_input_layer(topology[0], -1, number_of_threads, options.has_bias());
  _layers.emplace_back(std::move(layer));

  const auto& optimiser_type = options.optimiser_type();
  const auto& residual_layer_jump = options.residual_layer_jump();

  // then the hidden layers
  for (size_t layer_number = 1; layer_number < number_of_layers -1; ++layer_number)
  {
    const auto hidden_layer_number = layer_number - 1;
    const auto& layer_details = hidden_layers[hidden_layer_number];
    auto num_neurons_current_layer = topology[layer_number];
    auto num_neurons_next_layer = topology[layer_number + 1];
    auto dropout_rate = layer_details.get_dropout();
    const auto& previous_layer = *_layers.back();
    const auto residual_layer_number = compute_residual_layer(static_cast<int>(layer_number), residual_layer_jump);
    
    layer = create_hidden_layer(layer_details.get_weight_decay(), previous_layer, optimiser_type, residual_layer_number, dropout_rate, layer_details, number_of_threads, options.has_bias());

    _layers.emplace_back(std::move(layer));
  }

  // finally, the output layer
  const auto& output_layer_details = options.output_layer_details();
  layer = create_output_layer(topology.back(), *_layers.back(), output_layer_details, optimiser_type, number_of_threads, options.has_bias());

  _layers.emplace_back(std::move(layer));
}

Layers::~Layers()
{
  delete _update_weights_pool;
  _update_weights_pool = nullptr;
}

Layers::Layers(const Layers& src) noexcept :
  _update_weights_pool(nullptr)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  _update_weights_pool = new TaskQueuePool<void>(src._update_weights_pool->get_number_of_threads());
  _layers.reserve(src._layers.size());
  for(const auto& layer : src._layers)
  {
    _layers.emplace_back(std::unique_ptr<Layer>(layer->clone()));
  }
}

Layers::Layers(
  const NeuralNetworkOptions& options,
  const std::vector<std::unique_ptr<Layer>>& layers) noexcept :
  _update_weights_pool(nullptr)
{
  _update_weights_pool = new TaskQueuePool<void>(options.number_of_threads());
  _layers.reserve(layers.size());
  for (const auto& layer : layers)
  {
    _layers.emplace_back(std::unique_ptr<Layer>(layer->clone()));
  }
}

Layers::Layers(Layers&& src) noexcept :
  _layers(std::move(src._layers)),
  _training_gradients_buffer(std::move(src._training_gradients_buffer)),
  _training_hidden_states_buffer(std::move(src._training_hidden_states_buffer)),
  _update_weights_pool(nullptr)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");

  _update_weights_pool = src._update_weights_pool;
  src._update_weights_pool = nullptr;
}

Layers& Layers::operator=(const Layers& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  if(this != &src)
  {
    delete _update_weights_pool;
    _update_weights_pool = new TaskQueuePool<void>(src._update_weights_pool->get_number_of_threads());

    _layers.clear();
    _layers.reserve(src._layers.size());
    for(const auto& layer : src._layers)
    {
      _layers.emplace_back(std::unique_ptr<Layer>(layer->clone()));
    }
  }
  return *this;
}

Layers& Layers::operator=(Layers&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  if (this != &src)
  {
    _layers = std::move(src._layers);

    _training_gradients_buffer = std::move(src._training_gradients_buffer);
    _training_hidden_states_buffer = std::move(src._training_hidden_states_buffer);

    delete _update_weights_pool;
    _update_weights_pool = src._update_weights_pool;

    src._update_weights_pool = nullptr;
  }
  return *this;
}

const Layer& Layers::operator[](unsigned index ) const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
#if VALIDATE_DATA == 1
  if (index >= _layers.size())
  {
    Logger::panic("Layers trying to get an index past the size!");
  }
#endif
  return *_layers[index];
}

Layer& Layers::operator[](unsigned index )
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
#if VALIDATE_DATA ==1
  if (index >= _layers.size())
  {
    Logger::panic("Layers trying to get an index past the size!");
  }
#endif
  return *_layers[index];
}

const ResidualProjector* Layers::get_residual_layer_projector(unsigned index) const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
#if VALIDATE_DATA ==1
  if (index >= _layers.size())
  {
    Logger::panic("Layers trying to get a residual layer projector index past the size!");
  }
#endif
  return _layers[index]->get_residual_projector();
}

int Layers::get_residual_layer_number(unsigned index) const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
#if VALIDATE_DATA ==1
  if (index >= _layers.size())
  {
    Logger::panic("Layers trying to get a residual layer projector index past the size!");
  }
#endif
  return _layers[index]->get_residual_layer_number();
}

const std::vector<std::unique_ptr<Layer>>& Layers::get_layers() const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return _layers;
}

std::vector<std::unique_ptr<Layer>>& Layers::get_layers()
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return _layers;
}

int Layers::compute_residual_layer(int current_layer_index, int residual_layer_jump) const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  if(residual_layer_jump <= 0)
  {
    return -1;  // disabled
  }

  int residual_layer_index = current_layer_index - residual_layer_jump;
  if (residual_layer_index < 0) 
  {
      return -1;
  }
  return residual_layer_index;
}

std::unique_ptr<Layer> Layers::create_input_layer(unsigned num_neurons_in_this_layer, int residual_layer_number, int number_of_threads, bool has_bias)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return std::make_unique<FFLayer>(
    0, 
    0, 
    num_neurons_in_this_layer, 
    0.0,      // no weight decay
    Layer::LayerType::Input, 
    activation(activation::method::linear, 0.00),   //  Linear has no activation apha
    OptimiserType::None, 
    residual_layer_number, 
    0.0,      // no dropout for input layer
    nullptr,  // no residual projector for input
    number_of_threads,
    has_bias
  );
}

std::unique_ptr<Layer> Layers::create_hidden_layer(
  double weight_decay, 
  const Layer& previous_layer, 
  const OptimiserType& optimiser_type, 
  int residual_layer_number, 
  double dropout_rate, 
  const LayerDetails& layer_details,
  int number_of_threads,
  bool has_bias)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  unsigned layer_index = previous_layer.get_layer_index() + 1;

  unsigned num_neurons_in_this_layer = layer_details.get_size();

  switch (layer_details.get_type())
  {
  case LayerDetails::LayerType::Elman:
    return std::make_unique<ElmanRNNLayer>(
      layer_index, 
      previous_layer.get_number_neurons(), 
      num_neurons_in_this_layer, 
      weight_decay, 
      Layer::LayerType::Hidden, 
      layer_details.get_activation(),
      optimiser_type, 
      residual_layer_number, 
      dropout_rate,
      create_residual_projector(layer_details.get_activation(), residual_layer_number, num_neurons_in_this_layer, weight_decay),
      number_of_threads,
      has_bias);

  case LayerDetails::LayerType::Gru:
    return std::make_unique<GRURNNLayer>(
      layer_index,
      previous_layer.get_number_neurons(),
      num_neurons_in_this_layer,
      weight_decay,
      Layer::LayerType::Hidden,
      layer_details.get_activation(),
      optimiser_type,
      residual_layer_number,
      dropout_rate,
      create_residual_projector(layer_details.get_activation(), residual_layer_number, num_neurons_in_this_layer, weight_decay),
      number_of_threads,
      has_bias);

  case LayerDetails::LayerType::FF:
    return std::make_unique<FFLayer>(
      layer_index, 
      previous_layer.get_number_neurons(),
      num_neurons_in_this_layer, 
      weight_decay, 
      Layer::LayerType::Hidden, 
      layer_details.get_activation(),
      optimiser_type, 
      residual_layer_number,
      dropout_rate,
      create_residual_projector(layer_details.get_activation(), residual_layer_number, num_neurons_in_this_layer, weight_decay),
      number_of_threads,
      has_bias);

  default:
    Logger::panic("Unknown or unsupported layer type!");
  }
}

std::unique_ptr<Layer> Layers::create_output_layer(unsigned num_neurons_in_this_layer, const Layer& previous_layer, const std::vector<OutputLayerDetails>& output_layer_details, const OptimiserType& optimiser_type, int number_of_threads, bool has_bias)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  unsigned layer_index = previous_layer.get_layer_index() + 1;
  if (output_layer_details.size() == 1)
  {
    Logger::trace("Creating simple output layer.");
  }
  else
  {
    Logger::info("Creating compound output layer with ", output_layer_details.size(), " output layer details.");
  }
  
  // compound output
  return std::make_unique<FFOutputLayer>(
    layer_index, 
    output_layer_details,
    previous_layer.get_number_neurons(),
    num_neurons_in_this_layer, 
    optimiser_type, 
    number_of_threads,
    has_bias);
}

ResidualProjector* Layers::create_residual_projector(
  const activation& activation_method,
  int residual_layer_number,
  int number_of_neurons_in_current_layer,
  double weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  if (residual_layer_number < 0)
  {
    return nullptr;
  }

  auto number_of_neurons_in_that_layer = _layers[residual_layer_number]->get_number_neurons();

  return ResidualProjector::create(
    residual_layer_number,
    activation_method, 
    number_of_neurons_in_that_layer, 
    number_of_neurons_in_current_layer, 
    weight_decay);
}

void Layers::calculate_forward_feed(
  const NeuralNetworkOptions& options,
  std::vector<GradientsAndOutputs>& gradients_and_output,
  std::vector<std::vector<double>>::const_iterator inputs_begin,
  size_t batch_size,
  std::vector<HiddenStates>& hidden_states,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");

#if VALIDATE_DATA ==1
  if (gradients_and_output.size() < batch_size) /* can be less in case of mismatch total BPTTs */
  {
    Logger::panic("Layers trying calculate forward feed but output size does not match batch size!");
  }
  if (hidden_states.size() < batch_size)
  {
    Logger::panic("Layers trying calculate forward feed but hidden states size does not match batch size!");
  }
  if (hidden_states.size() != gradients_and_output.size())
  {
    Logger::panic("Layers trying calculate forward feed but hidden states size does not match gradients and output!");
  }

#endif

  std::shared_lock<std::shared_mutex> read(_mutex);

  // --- 1. Store input layer outputs for the entire batch ---
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& current_input = *(inputs_begin + b);
    const size_t input_size = input_layer().get_number_neurons();

    if (current_input.size() == input_size)
    {
      gradients_and_output[b].set_outputs(0, current_input);
      if (options.enable_bptt() && options.bptt_max_ticks() > 1)
      {
        std::vector<double> expanded;
        const int ticks = options.bptt_max_ticks();
        expanded.reserve(input_size * ticks);
        for (int t = 0; t < ticks; ++t) expanded.insert(expanded.end(), current_input.begin(), current_input.end());
        gradients_and_output[b].set_rnn_outputs(0, expanded);
      }
    }
    else if (options.enable_bptt() && input_size > 0 && current_input.size() % input_size == 0)
    {
      // Sequence input provided!
      // Set the standard output to the LAST time step (so strict topology checks pass)
      std::vector<double> last_step(current_input.end() - input_size, current_input.end());
      gradients_and_output[b].set_outputs(0, last_step);

      // Set the full sequence for RNN layers to consume
      gradients_and_output[b].set_rnn_outputs(0, current_input);
    }
    else
    {
      // Fallback (will likely assert if size mismatch)
      gradients_and_output[b].set_outputs(0, current_input);
    }
  }

  // --- 2. Forward propagate layer by layer for the entire batch ---
  for (size_t layer_number = 1; layer_number < size(); ++layer_number)
  {
    const auto& previous_layer = layer(static_cast<unsigned>(layer_number - 1));
    const auto& current_layer = layer(static_cast<unsigned>(layer_number));

    // Prepare batched residual outputs if needed
    std::vector<std::vector<double>> batch_residual_values;
    const auto* residual_projector = get_residual_layer_projector(static_cast<unsigned>(layer_number));
    if (residual_projector != nullptr)
    {
      auto residual_layer_number = get_residual_layer_number(static_cast<unsigned>(layer_number));
      std::vector<std::vector<double>> batch_residual_inputs;
      batch_residual_values.reserve(batch_size);
      for (size_t b = 0; b < batch_size; ++b)
      {
        batch_residual_inputs.emplace_back(gradients_and_output[b].get_outputs(static_cast<unsigned>(residual_layer_number)));
      }
      batch_residual_values = residual_projector->project_batch(batch_residual_inputs);
    }

    // Ensure hidden state vectors are sized correctly
    for (size_t b = 0; b < batch_size; ++b)
    {
      if (current_layer.use_bptt())
      {
        std::vector<double> prev_rnn_out = gradients_and_output[b].get_rnn_outputs(previous_layer.get_layer_index());
        if (prev_rnn_out.empty()) prev_rnn_out = gradients_and_output[b].get_outputs(previous_layer.get_layer_index());
        const size_t n_prev = previous_layer.get_number_neurons();
        const size_t num_time_steps = n_prev > 0 ? prev_rnn_out.size() / n_prev : 0;
        hidden_states[b].at(layer_number).assign(num_time_steps, HiddenState(current_layer.get_number_neurons()));
      }
      else
      {
        hidden_states[b].at(layer_number).assign(1, HiddenState(current_layer.get_number_neurons()));
      }
    }

    // Call batched forward feed
    current_layer.calculate_forward_feed(
      gradients_and_output,
      previous_layer,
      batch_residual_values,
      hidden_states,
      batch_size,
      is_training
    );
  }
}

void Layers::calculate_back_propagation(
  const NeuralNetworkOptions& options,
  std::vector<GradientsAndOutputs>& gradients,
  std::vector<std::vector<double>>::const_iterator outputs_begin,
  size_t batch_size,
  const std::vector<HiddenStates>& hidden_states) const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");

  calculate_back_propagation_output_layer(options, gradients, outputs_begin, batch_size, hidden_states);
  calculate_back_propagation_hidden_layers(options, gradients, batch_size, hidden_states);
  calculate_back_propagation_input_layer(options, gradients, batch_size);
}

void Layers::calculate_back_propagation_input_layer(
  const NeuralNetworkOptions& options,
  std::vector<GradientsAndOutputs>& gradients,
  size_t batch_size) const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");

  // input layer is all 0, (bias is included)
  const auto& input_gradients = std::vector<double>(input_layer().get_number_neurons(), 0.0);
  for (size_t i = 0; i < batch_size; ++i)
  {
    gradients[i].set_gradients(0, input_gradients);
  }
}

void Layers::calculate_back_propagation_output_layer(
  const NeuralNetworkOptions& options,
  std::vector<GradientsAndOutputs>& gradients,
  std::vector<std::vector<double>>::const_iterator outputs_begin,
  size_t batch_size,
  const std::vector<HiddenStates>& hidden_states) const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  const auto& output_layer_number = static_cast<unsigned>(size() - 1);
  output_layer().calculate_output_gradients(gradients, outputs_begin, hidden_states, batch_size);
}

void Layers::calculate_back_propagation_hidden_layers(
  const NeuralNetworkOptions& options,
  std::vector<GradientsAndOutputs>& gradients,
  size_t batch_size,
  const std::vector<HiddenStates>& hidden_states) const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  // we are going backward from output to input
  for (auto layer_number = (int)size() - 2; layer_number > 0; --layer_number)
  {
    auto& hidden_0 = layer(static_cast<unsigned>(layer_number));
    const auto& hidden_1 = layer(static_cast<unsigned>(layer_number + 1));

    std::vector<std::vector<double>> batch_next_gradients;
    batch_next_gradients.reserve(batch_size);
    for (size_t b = 0; b < batch_size; ++b)
    {
      const auto& g = gradients[b];
      std::vector<double> grad;
      if (options.enable_bptt() && hidden_1.use_bptt())
      {
        grad = g.get_rnn_gradients(static_cast<unsigned>(layer_number + 1));
      }
      if (grad.empty())
      {
        grad = g.get_gradients(static_cast<unsigned>(layer_number + 1));
      }
      batch_next_gradients.emplace_back(std::move(grad));
    }

    hidden_0.calculate_hidden_gradients(gradients, hidden_1, batch_next_gradients, hidden_states, batch_size, options.bptt_max_ticks());
  }
}

void Layers::update_weights(
  const NeuralNetworkOptions& options,
  const std::vector<GradientsAndOutputs>& batch_gradients,
  double learning_rate,
  size_t batch_size,
  const std::vector<HiddenStates>& hidden_states)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");

  if (batch_size == 0)
  {
    return;
  }

  // 1. Have each layer calculate and store its own gradients
  for (unsigned i = 1; i < size(); ++i)
  {
    _update_weights_pool->enqueue(
      [i, &batch_gradients, &hidden_states, batch_size, &options, this]()
      {
        auto& layer_a = *_layers.at(i);
        auto& layer_b = *_layers.at(i -1);
        layer_a.calculate_and_store_gradients(batch_gradients, hidden_states, layer_b, batch_size, options.bptt_max_ticks());
      });
  }
  _update_weights_pool->get();

  // 2. Calculate global gradient norm for clipping
  std::vector<double> layer_norms(size(), 0.0);
  for (unsigned i = 1; i < size(); ++i)
  {
    _update_weights_pool->enqueue(
      [i, &layer_norms, this]()
      {
        layer_norms[i] = _layers[i]->get_gradient_norm_sq();
      });
  }
  _update_weights_pool->get();

  double total_norm_sq = 0.0;
  for (double norm : layer_norms)
  {
    total_norm_sq += norm;
  }

  double clipping_scale = 1.0;
  const double gradient_clip_threshold = options.clip_threshold();
  if (gradient_clip_threshold > 0.0 && total_norm_sq > 0.0)
  {
    const double norm = std::sqrt(total_norm_sq);
    if (norm > gradient_clip_threshold)
    {
      clipping_scale = gradient_clip_threshold / norm;
    }
  }

  // 3. Apply the stored (and now clipped) gradients
  std::unique_lock<std::shared_mutex> write(_mutex);
  for (unsigned i = 1; i < size(); ++i)
  {
    _update_weights_pool->enqueue([i, learning_rate, clipping_scale, this]()
      {
        auto& layer_a = *_layers[i];
        layer_a.apply_stored_gradients(learning_rate, clipping_scale);
      });
  }
  _update_weights_pool->get();
}

std::vector<std::vector<double>> Layers::think(const NeuralNetworkOptions& options, const std::vector<std::vector<double>>& inputs) const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  const size_t batch_size = inputs.size();
  if (batch_size == 0) return {};

  std::vector<GradientsAndOutputs> batch_gradients;
  batch_gradients.resize(batch_size, GradientsAndOutputs(options.topology()));
  std::vector<HiddenStates> hidden_states;
  hidden_states.resize(batch_size, HiddenStates(options.topology()));

  calculate_forward_feed(options, batch_gradients, inputs.begin(), batch_size, hidden_states, false);

  std::vector<std::vector<double>> outputs;
  outputs.reserve(batch_size);
  for (size_t i = 0; i < batch_size; ++i)
  {
    outputs.push_back(batch_gradients[i].output_back());
  }
  return outputs;
}

std::vector<double> Layers::think(const NeuralNetworkOptions& options, const std::vector<double>& inputs) const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");

  std::vector<GradientsAndOutputs> gradients;
  gradients.push_back(GradientsAndOutputs(options.topology()));
  std::vector<HiddenStates> hidden_states;
  hidden_states.resize(1, HiddenStates(options.topology()));

  const std::vector<std::vector<double>> all_inputs = { inputs };
  calculate_forward_feed(options, gradients, all_inputs.begin(), 1, hidden_states, false);
  return gradients.front().output_back();
}

void Layers::train(
  const NeuralNetworkOptions& options,
  const double learning_rate,
  std::vector<std::vector<double>>::const_iterator& inputs_begin,
  std::vector<std::vector<double>>::const_iterator& outputs_begin,
  const size_t batch_size)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");

  if (_training_gradients_buffer.size() < batch_size)
  {
    _training_gradients_buffer.resize(batch_size, GradientsAndOutputs(options.topology()));
  }
  if (_training_hidden_states_buffer.size() < batch_size)
  {
    _training_hidden_states_buffer.resize(batch_size, HiddenStates(options.topology()));
  }

  // 2. Calculate gradients via back-propagation
  // Note: calculate_forward_feed and calculate_back_propagation now fully manage their buffers.
  calculate_forward_feed(options, _training_gradients_buffer, inputs_begin, batch_size, _training_hidden_states_buffer, true);
  calculate_back_propagation(options, _training_gradients_buffer, outputs_begin, batch_size, _training_hidden_states_buffer);
  update_weights(options, _training_gradients_buffer, learning_rate, batch_size, _training_hidden_states_buffer);
}

