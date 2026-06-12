#include "elmanrnnlayer.h"
#include "fflayer.h"
#include "ffoutputlayer.h"
#include "grurnnlayer.h"
#include "layer.h"
#include "layers.h"
#include "lstmlayer.h"
#include "multioutputlayer.h"


namespace myoddweb::nn
{
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

  const auto& residual_layer_jump = options.residual_layer_jump();

  // then the hidden layers
  for (size_t layer_number = 1; layer_number < number_of_layers -1; ++layer_number)
  {
    const auto hidden_layer_number = layer_number - 1;
    const auto& layer_details = hidden_layers[hidden_layer_number];
    const auto& previous_layer = *_layers.back();
    const auto residual_layer_number = compute_residual_layer(static_cast<int>(layer_number), residual_layer_jump);
    
    layer = create_hidden_layer(previous_layer, residual_layer_number, layer_details, number_of_threads, options.has_bias());

    _layers.emplace_back(std::move(layer));
  }

  // finally, the output layer
  if (options.has_multi_output())
  {
    layer = create_multi_output_layer(topology.back(), *_layers.back(), options.multi_output_layer_details(), number_of_threads, options.has_bias());
  }
  else
  {
    const auto& output_layer_details = options.output_layer_details();
    layer = create_output_layer(topology.back(), *_layers.back(), output_layer_details, number_of_threads, options.has_bias());
  }

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
  std::shared_lock<std::shared_mutex> lock(src._mutex);
  _update_weights_pool = new TaskQueuePool<void>(src._update_weights_pool->get_number_of_threads());
  _layers.reserve(src._layers.size());
  for(const auto& layer : src._layers)
  {
    _layers.emplace_back(std::unique_ptr<Layer>(layer->clone()));
  }
  _training_gradients_buffer = src._training_gradients_buffer;
  _training_hidden_states_buffer = src._training_hidden_states_buffer;
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
    std::unique_lock<std::shared_mutex> lhs_lock(_mutex, std::defer_lock);
    std::shared_lock<std::shared_mutex> rhs_lock(src._mutex, std::defer_lock);
    std::lock(lhs_lock, rhs_lock);

    delete _update_weights_pool;
    _update_weights_pool = new TaskQueuePool<void>(src._update_weights_pool->get_number_of_threads());

    _layers.clear();
    _layers.reserve(src._layers.size());
    for(const auto& layer : src._layers)
    {
      _layers.emplace_back(std::unique_ptr<Layer>(layer->clone()));
    }

    _training_gradients_buffer = src._training_gradients_buffer;
    _training_hidden_states_buffer = src._training_hidden_states_buffer;
  }
  return *this;
}

Layers& Layers::operator=(Layers&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  if (this != &src)
  {
    std::unique_lock<std::shared_mutex> lhs_lock(_mutex, std::defer_lock);
    std::unique_lock<std::shared_mutex> rhs_lock(src._mutex, std::defer_lock);
    std::lock(lhs_lock, rhs_lock);

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

double Layers::get_temperature(unsigned output_layer_index) const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return output_layer().get_temperature(output_layer_index);
}

double Layers::get_inference_temperature(unsigned output_layer_index) const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return output_layer().get_inference_temperature(output_layer_index);
}

void Layers::set_inference_temperature(unsigned output_layer_index, double t) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  _layers.back()->set_inference_temperature(output_layer_index, t);
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
    Layer::Role::Input,
    activation(activation::method::linear, 0.00),   //  Linear has no activation apha
    OptimiserType::None, 
    residual_layer_number, 
    0.0,      // no dropout for input layer
    nullptr,  // no residual projector for input
    number_of_threads,
    has_bias,
    0.0
  );
}

std::unique_ptr<Layer> Layers::create_hidden_layer(
  const Layer& previous_layer,
  int residual_layer_number,
  const LayerDetails& layer_details,
  int number_of_threads,
  bool has_bias)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  unsigned layer_index = previous_layer.get_layer_index() + 1;
  unsigned num_neurons_in_this_layer = layer_details.get_size();
  double weight_decay = layer_details.get_weight_decay();

  return Layer::create_hidden_layer(
    layer_index,
    previous_layer.get_number_neurons(),
    layer_details,
    number_of_threads,
    has_bias,
    residual_layer_number,
    create_residual_projector(layer_details.get_activation(), residual_layer_number, num_neurons_in_this_layer, weight_decay)
  );
}
std::unique_ptr<Layer> Layers::create_output_layer(unsigned num_neurons_in_this_layer, const Layer& previous_layer, const std::vector<OutputLayerDetails>& output_layer_details, int number_of_threads, bool has_bias)
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
    number_of_threads,
    has_bias);
}

std::unique_ptr<Layer> Layers::create_multi_output_layer(unsigned num_neurons_in_this_layer, const Layer& previous_layer, const std::vector<MultiOutputLayerDetails>& multi_output_layer_details, int number_of_threads, bool has_bias)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  unsigned layer_index = previous_layer.get_layer_index() + 1;
  return std::make_unique<MultiOutputLayer>(
    layer_index,
    previous_layer.get_number_neurons(),
    num_neurons_in_this_layer,
    multi_output_layer_details,
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
      batch_residual_inputs.reserve(batch_size);
      for (size_t b = 0; b < batch_size; ++b)
      {
        const auto src_span = gradients_and_output[b].get_outputs(static_cast<unsigned>(residual_layer_number));
        batch_residual_inputs.emplace_back(src_span.begin(), src_span.end());
      }
      batch_residual_values = residual_projector->project_batch(batch_residual_inputs);
    }

    for (size_t b = 0; b < batch_size; ++b)
    {
      const auto prev_rnn_span = gradients_and_output[b].get_rnn_outputs(previous_layer.get_layer_index());
      const auto prev_std_span = gradients_and_output[b].get_outputs(previous_layer.get_layer_index());

      const size_t seq_size = !prev_rnn_span.empty() ? prev_rnn_span.size() : prev_std_span.size();
      const size_t n_prev = previous_layer.get_number_neurons();
      const size_t num_time_steps = (n_prev > 0 && !prev_rnn_span.empty()) ? seq_size / n_prev : 1;
      
      hidden_states[b].assign(layer_number, num_time_steps, HiddenState(), current_layer.get_pre_activation_multiplier());
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
  (void)options;
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
  auto& ol = output_layer();
  ol.calculate_output_gradients(gradients, outputs_begin, hidden_states, batch_size);
  
  if (const auto& branched = dynamic_cast<const MultiOutputLayer*>(&ol))
  {
    branched->backprop_branches(batch_size, options.bptt_max_ticks());
  }
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
    
    // Check if next layer is Branched
    if (auto branched = dynamic_cast<const MultiOutputLayer*>(&hidden_1)) {
       // Only call backprop_branches if it's NOT the output layer, 
       // because calculate_back_propagation_output_layer already did it.
       if (static_cast<unsigned>(layer_number + 1) != size() - 1)
       {
         branched->backprop_branches(batch_size, options.bptt_max_ticks());
       }
       batch_next_gradients = branched->get_trunk_gradients(batch_size);
       hidden_0.calculate_hidden_gradients_from_output_gradients(gradients, batch_next_gradients, hidden_states, batch_size, options.bptt_max_ticks());
    } else {
       bool next_is_recurrent = false;
       batch_next_gradients.reserve(batch_size);
       for (size_t b = 0; b < batch_size; ++b)
       {
         const auto& g = gradients[b];
         std::vector<double> grad;
         
         const auto rnn_span = g.get_rnn_gradients(static_cast<unsigned>(layer_number + 1));
         if (!rnn_span.empty())
         {
           grad.assign(rnn_span.begin(), rnn_span.end());
           next_is_recurrent = true;
         }
         
         if (grad.empty())
         {
           const auto std_span = g.get_gradients(static_cast<unsigned>(layer_number + 1));
           grad.assign(std_span.begin(), std_span.end());
         }
         batch_next_gradients.emplace_back(std::move(grad));
       }
       if (next_is_recurrent)
       {
         hidden_0.calculate_hidden_gradients_from_output_gradients(gradients, batch_next_gradients, hidden_states, batch_size, options.bptt_max_ticks());
       }
       else
       {
         hidden_0.calculate_hidden_gradients(gradients, hidden_1, batch_next_gradients, hidden_states, batch_size, options.bptt_max_ticks());
       }
    }
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

  const auto& num_threads = _update_weights_pool->get_number_of_threads();
  if (num_threads <= 1)
  {
    // 1. Have each layer calculate and store its own gradients
    for (unsigned i = 1; i < size(); ++i)
    {
      auto& layer_a = *_layers.at(i);
      auto& layer_b = *_layers.at(i - 1);
      layer_a.calculate_and_store_gradients(batch_gradients, hidden_states, layer_b, batch_size, options.bptt_max_ticks());
    }
  }
  else
  {
    // 1. Have each layer calculate and store its own gradients
    for (unsigned i = 1; i < size(); ++i)
    {
      _update_weights_pool->enqueue(
        [i, &batch_gradients, &hidden_states, batch_size, &options, this]()
        {
          auto& layer_a = *_layers.at(i);
          auto& layer_b = *_layers.at(i - 1);
          layer_a.calculate_and_store_gradients(batch_gradients, hidden_states, layer_b, batch_size, options.bptt_max_ticks());
        });
    }
    _update_weights_pool->get();
  }

  // 2. Calculate global gradient norm for clipping
  std::vector<double> layer_norms(size(), 0.0);
  if (num_threads <= 1)
  {
    for (unsigned i = 1; i < size(); ++i)
    {
      layer_norms[i] = _layers[i]->get_gradient_norm_sq();
    }
  }
  else
  {
    for (unsigned i = 1; i < size(); ++i)
    {
      _update_weights_pool->enqueue(
        [i, &layer_norms, this]()
        {
          layer_norms[i] = _layers[i]->get_gradient_norm_sq();
        });
    }
    _update_weights_pool->get();
  }

  double total_norm_sq = 0.0;
  for (double norm : layer_norms)
  {
    total_norm_sq += norm;
  }
  const double total_norm = std::sqrt(total_norm_sq);

  double clipping_scale = 1.0;
  const double gradient_clip_threshold = options.clip_threshold();
  if (gradient_clip_threshold > 0.0 && total_norm > 0.0)
  {
    if (total_norm > gradient_clip_threshold)
    {
      clipping_scale = gradient_clip_threshold / total_norm;
    }
  }

  if (!std::isfinite(total_norm)) 
  {
      Logger::panic("CRITICAL: Explosive gradients detected (norm is NaN/Inf)!");
  }

  // 3. Apply the stored (and now clipped) gradients
  std::unique_lock<std::shared_mutex> write(_mutex);
  if (num_threads <= 1)
  {
    for (unsigned i = 1; i < size(); ++i)
    {
      auto& layer_a = *_layers[i];
      layer_a.apply_stored_gradients(learning_rate, clipping_scale);
    }
  }
  else
  {
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
    _training_gradients_buffer.reserve(batch_size);
    while (_training_gradients_buffer.size() < batch_size)
    {
      _training_gradients_buffer.emplace_back(options.topology());
    }
  }
  if (_training_hidden_states_buffer.size() < batch_size)
  {
    _training_hidden_states_buffer.reserve(batch_size);
    while (_training_hidden_states_buffer.size() < batch_size)
    {
      _training_hidden_states_buffer.emplace_back(options.topology());
    }
  }

  // Zero out the items we are about to use
  for (size_t i = 0; i < batch_size; ++i)
  {
    _training_gradients_buffer[i].zero();
    _training_hidden_states_buffer[i].zero();
  }

  // 2. Calculate gradients via back-propagation
  calculate_forward_feed(options, _training_gradients_buffer, inputs_begin, batch_size, _training_hidden_states_buffer, true);
  calculate_back_propagation(options, _training_gradients_buffer, outputs_begin, batch_size, _training_hidden_states_buffer);
  update_weights(options, _training_gradients_buffer, learning_rate, batch_size, _training_hidden_states_buffer);
  cache_recurrent_weights();
}

void Layers::cache_recurrent_weights()
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  for (auto& layer : _layers)
  {
    layer->cache_recurrent_weights();
  }
}

} // namespace myoddweb::nn
