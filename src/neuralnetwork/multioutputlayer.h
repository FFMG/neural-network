#pragma once

#include "elmanrnnlayer.h"
#include "fflayer.h"
#include "ffoutputlayer.h"
#include "grurnnlayer.h"
#include "layer.h"
#include "layerdetails.h"
#include "multioutputlayerdetails.h"
#include "outputlayer.h"
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>

/**
 * A minimal layer implementation to act as a source for branch layers.
 * Branch layers expect their input from index 0 of the branch's internal buffers.
 */
class MultiInputProxyLayer final : public Layer
{
public:
  MultiInputProxyLayer(unsigned num_neurons) :
    Layer(0, Layer::LayerType::Input, activation(activation::method::linear, 0.0), OptimiserType::None, -1, num_neurons, num_neurons, {}, false, 0.0, nullptr, 1, 0.0)
  {
    MYODDWEB_PROFILE_FUNCTION("MultiInputProxyLayer");
  }

  Layer* clone() const override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiInputProxyLayer");
    return new MultiInputProxyLayer(*this);
  }

  void calculate_forward_feed(std::vector<GradientsAndOutputs>&, const Layer&, const std::vector<std::vector<double>>&, std::vector<HiddenStates>&, size_t, bool) const override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiInputProxyLayer");
  }
  void calculate_output_gradients(std::vector<GradientsAndOutputs>&, std::vector<std::vector<double>>::const_iterator, const std::vector<HiddenStates>&, size_t) const override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiInputProxyLayer");
  }
  void calculate_hidden_gradients(std::vector<GradientsAndOutputs>&, const Layer&, const std::vector<std::vector<double>>&, const std::vector<HiddenStates>&, size_t, int) const override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiInputProxyLayer");
  }
  void calculate_and_store_gradients(const std::vector<GradientsAndOutputs>&, const std::vector<HiddenStates>&, const Layer&, size_t, int) override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiInputProxyLayer");
  }
  void apply_stored_gradients(double, double) override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiInputProxyLayer");
  }
  double get_gradient_norm_sq() const override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiInputProxyLayer");
    return 0.0;
  }
};

class MultiOutputLayer final : public Layer, public OutputLayer
{
public:
  struct Branch
  {
    std::vector<std::unique_ptr<Layer>> layers;
    // Internal buffers for this branch
    std::vector<GradientsAndOutputs> gradients_and_outputs;
    std::vector<HiddenStates> hidden_states;
    std::vector<unsigned> topology;
    
    Branch()
    {
      MYODDWEB_PROFILE_FUNCTION("Branch");
    }

    Branch(const Branch& src)
    {
      MYODDWEB_PROFILE_FUNCTION("Branch");
      for (const auto& l : src.layers)
      {
        layers.emplace_back(l->clone());
      }
      topology = src.topology;
    }

    Branch(Branch&& src) noexcept : 
      layers(std::move(src.layers)),
      gradients_and_outputs(std::move(src.gradients_and_outputs)),
      hidden_states(std::move(src.hidden_states)),
      topology(std::move(src.topology))
    {
      MYODDWEB_PROFILE_FUNCTION("Branch");
    }

    Branch& operator=(const Branch& src)
    {
      MYODDWEB_PROFILE_FUNCTION("Branch");
      if (this != &src)
      {
        layers.clear();
        for (const auto& l : src.layers)
        {
          layers.emplace_back(l->clone());
        }
        topology = src.topology;
      }
      return *this;
    }

    Branch& operator=(Branch&& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("Branch");
      if (this != &src)
      {
        layers = std::move(src.layers);
        gradients_and_outputs = std::move(src.gradients_and_outputs);
        hidden_states = std::move(src.hidden_states);
        topology = std::move(src.topology);
      }
      return *this;
    }

    void init_buffers(size_t batch_size)
    {
      MYODDWEB_PROFILE_FUNCTION("Branch");
      if (gradients_and_outputs.size() < batch_size)
      {
        gradients_and_outputs.resize(batch_size, GradientsAndOutputs(topology));
        hidden_states.resize(batch_size, HiddenStates(topology));
      }
      for(size_t i=0; i<batch_size; ++i)
      {
        gradients_and_outputs[i].zero();
        hidden_states[i].zero();
      }
    }
  };

  MultiOutputLayer(
    unsigned layer_index,
    unsigned num_inputs,
    unsigned num_outputs,
    const std::vector<MultiOutputLayerDetails>& multi_output_layer_details,
    int number_of_threads,
    bool has_bias
  ) :
    Layer(layer_index, Layer::LayerType::Branched, layer_activation_helper(activation(activation::method::linear, 0.0), num_inputs, num_outputs), OptimiserType::None, -1, {}, has_bias, Layer::create_w_decays(num_inputs, num_outputs, 0.0), nullptr, number_of_threads, 0.0),
    OutputLayer(extract_output_details(multi_output_layer_details))
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    
    for (const auto& multi_output_layer_detail : multi_output_layer_details)
    {
      Branch branch;
      unsigned prev_n = num_inputs;
      branch.topology.push_back(num_inputs); // index 0 is input to branch
      
      // Hidden layers in branch
      for (size_t i = 0; i < multi_output_layer_detail.get_hidden_layers().size(); ++i)
      {
        const auto& ld = multi_output_layer_detail.get_hidden_layer(static_cast<unsigned>(i));
        auto l = std::make_unique<FFLayer>(
          (unsigned)branch.layers.size() + 1, // index in branch starts at 1
          prev_n,
          ld.get_size(),
          ld.get_weight_decay(),
          Layer::LayerType::Hidden,
          ld.get_activation(),
          ld.get_optimiser_type(),
          -1,
          ld.get_dropout(),
          nullptr,
          number_of_threads,
          has_bias,
          ld.get_momentum()
        );
        prev_n = ld.get_size();
        branch.layers.emplace_back(std::move(l));
        branch.topology.push_back(prev_n);
      }
      
      // Output layer in branch
      std::vector<OutputLayerDetails> olds = { multi_output_layer_detail.get_output_details()};
      auto ol = std::make_unique<FFOutputLayer>(
        (unsigned)branch.layers.size() + 1,
        olds,
        prev_n,
        multi_output_layer_detail.get_output_details().get_size(),
        number_of_threads,
        has_bias
      );
      branch.layers.emplace_back(std::move(ol));
      branch.topology.push_back(multi_output_layer_detail.get_output_details().get_size());
      _branches.emplace_back(std::move(branch));
    }
  }

  MultiOutputLayer(const MultiOutputLayer& src) :
    Layer(src),
    OutputLayer(src),
    _branches(src._branches)
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
  }

  MultiOutputLayer(MultiOutputLayer&& src) noexcept :
    Layer(std::move(src)),
    OutputLayer(std::move(src)),
    _branches(std::move(src._branches))
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
  }

  MultiOutputLayer& operator=(const MultiOutputLayer&) = delete;
  MultiOutputLayer& operator=(MultiOutputLayer&&) = delete;

  virtual ~MultiOutputLayer()
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
  }

  Layer* clone() const override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    return new MultiOutputLayer(*this);
  }

  void calculate_forward_feed(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& previous_layer,
    const std::vector<std::vector<double>>& /*batch_residual_output_values*/,
    std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    bool is_training) const override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    std::lock_guard<std::mutex> lock(_mutex);
    
    // 1. Initialize branch buffers
    for (auto& branch : const_cast<std::vector<Branch>&>(_branches))
    {
      branch.init_buffers(batch_size);
    }

    // 2. Set branch inputs from the trunk output
    const unsigned trunk_layer_index = previous_layer.get_layer_index();
    for (size_t b = 0; b < batch_size; ++b)
    {
       const auto trunk_output_span = batch_gradients_and_outputs[b].get_outputs(trunk_layer_index);
       std::vector<double> trunk_output(trunk_output_span.begin(), trunk_output_span.end());
       for (size_t i = 0; i < _branches.size(); ++i)
       {
         const_cast<Branch&>(_branches[i]).gradients_and_outputs[b].set_outputs(0, trunk_output);
       }
    }

    // 3. Forward through each branch (using a proxy for the trunk to match internal indices)
    MultiInputProxyLayer proxy(get_number_input_neurons());
    for (size_t i = 0; i < _branches.size(); ++i)
    {
      auto& branch = const_cast<Branch&>(_branches[i]);
      for (size_t l_idx = 0; l_idx < branch.layers.size(); ++l_idx)
      {
        const auto& current_l = *branch.layers[l_idx];
        const Layer& prev_l = (l_idx == 0) ? static_cast<const Layer&>(proxy) : *branch.layers[l_idx-1];

        // Ensure hidden state vectors are sized correctly for branch layers
        for (size_t b = 0; b < batch_size; ++b)
        {
          if (current_l.use_bptt())
          {
            const auto& prev_rnn_out = branch.gradients_and_outputs[b].get_rnn_outputs(prev_l.get_layer_index());
            const auto prev_std_out = branch.gradients_and_outputs[b].get_outputs(prev_l.get_layer_index());
            const size_t seq_size = !prev_rnn_out.empty() ? prev_rnn_out.size() : prev_std_out.size();
            const size_t n_prev = prev_l.get_number_neurons();
            const size_t num_time_steps = n_prev > 0 ? seq_size / n_prev : 0;
            branch.hidden_states[b].at(current_l.get_layer_index()).assign(num_time_steps, HiddenState(current_l.get_number_neurons()));
          }
          else
          {
            branch.hidden_states[b].at(current_l.get_layer_index()).assign(1, HiddenState(current_l.get_number_neurons()));
          }
        }

        current_l.calculate_forward_feed(
          branch.gradients_and_outputs,
          prev_l,
          {}, // No residuals in branches for now
          branch.hidden_states,
          batch_size,
          is_training
        );
      }
    }

    // 4. Concatenate branch outputs back to the main batch
    const unsigned this_layer_index = get_layer_index();
    for (size_t b = 0; b < batch_size; ++b)
    {
       std::vector<double> concatenated_output;
       concatenated_output.reserve(get_number_neurons());
       for (size_t i = 0; i < _branches.size(); ++i)
       {
         const auto b_out_span = _branches[i].gradients_and_outputs[b].output_back();
         concatenated_output.insert(concatenated_output.end(), b_out_span.begin(), b_out_span.end());
       }
       batch_gradients_and_outputs[b].set_outputs(this_layer_index, concatenated_output);
    }
  }

  void calculate_output_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    std::vector<std::vector<double>>::const_iterator target_outputs_begin,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size) const override
  {
     MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
     std::lock_guard<std::mutex> lock(_mutex);
     unsigned offset = 0;
     for (size_t i = 0; i < _branches.size(); ++i)
     {
       auto& branch = const_cast<Branch&>(_branches[i]);
       const unsigned b_out_size = branch.layers.back()->get_number_neurons();
       
       std::vector<std::vector<double>> sub_targets(batch_size);
       for(size_t b=0; b<batch_size; ++b)
       {
         const auto& full_target = *(target_outputs_begin + b);
         sub_targets[b].assign(full_target.begin() + offset, full_target.begin() + offset + b_out_size);
       }
       
       branch.layers.back()->calculate_output_gradients(
         branch.gradients_and_outputs,
         sub_targets.begin(),
         branch.hidden_states,
         batch_size
       );
       
       offset += b_out_size;
     }
  }

  // New method for custom backprop
  void backprop_branches(size_t batch_size, int bptt_max_ticks) const
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto& branch : const_cast<std::vector<Branch>&>(_branches))
    {
      for (int l_idx = (int)branch.layers.size() - 2; l_idx >= 0; --l_idx)
      {
        auto& current = *branch.layers[l_idx];
        const auto& next = *branch.layers[l_idx+1];
        
        std::vector<std::vector<double>> batch_next_gradients;
        batch_next_gradients.reserve(batch_size);
        for(size_t b=0; b<batch_size; ++b)
        {
          const auto g_span = branch.gradients_and_outputs[b].get_gradients(next.get_layer_index());
          batch_next_gradients.emplace_back(g_span.begin(), g_span.end());
        }

        current.calculate_hidden_gradients(
          branch.gradients_and_outputs,
          next,
          batch_next_gradients,
          branch.hidden_states,
          batch_size,
          bptt_max_ticks
        );
      }
    }
  }

  std::vector<std::vector<double>> get_trunk_gradients(size_t batch_size) const
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    std::lock_guard<std::mutex> lock(_mutex);
    std::vector<std::vector<double>> trunk_grads(batch_size, std::vector<double>(get_number_input_neurons(), 0.0));
    
    for (const auto& branch : _branches)
    {
      // Since first_layer.calculate_hidden_gradients was called, 
      // the gradients at index 0 of branch.gradients_and_outputs[b] 
      // contain the gradients for the branch input (the trunk).
      
      for(size_t b=0; b<batch_size; ++b)
      {
        const auto g_span = branch.gradients_and_outputs[b].get_gradients(0);
        for(size_t j=0; j<trunk_grads[b].size(); ++j)
        {
          trunk_grads[b][j] += g_span[j];
        }
      }
    }
    return trunk_grads;
  }

  void calculate_hidden_gradients(
    std::vector<GradientsAndOutputs>& /*batch_gradients_and_outputs*/,
    const Layer& /*next_layer*/,
    const std::vector<std::vector<double>>& /*batch_next_grad_matrix*/,
    const std::vector<HiddenStates>& /*batch_hidden_states*/,
    size_t /*batch_size*/,
    int /*bptt_max_ticks*/) const override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    // This is handled by backprop_branches + get_trunk_gradients in Layers.cpp
  }

  void calculate_and_store_gradients(
    const std::vector<GradientsAndOutputs>& /*batch_gradients_and_outputs*/,
    const std::vector<HiddenStates>& /*hidden_states*/,
    const Layer& previous_layer,
    size_t batch_size,
    int bptt_max_ticks) override
  {
     MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
     std::lock_guard<std::mutex> lock(_mutex);
     MultiInputProxyLayer proxy(get_number_input_neurons());
     for(auto& branch : _branches)
     {
       for(size_t l_idx = 0; l_idx < branch.layers.size(); ++l_idx)
       {
         auto& current_l = *branch.layers[l_idx];
         const Layer& prev_l = (l_idx == 0) ? static_cast<const Layer&>(proxy) : *branch.layers[l_idx-1];
         
         current_l.calculate_and_store_gradients(
           branch.gradients_and_outputs,
           branch.hidden_states,
           prev_l,
           batch_size,
           bptt_max_ticks
         );
       }
     }
  }

  void apply_stored_gradients(double learning_rate, double clipping_scale) override
  {
     MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
     std::lock_guard<std::mutex> lock(_mutex);
     for(auto& branch : _branches)
     {
       for(auto& layer : branch.layers)
       {
         layer->apply_stored_gradients(learning_rate, clipping_scale);
       }
     }
  }

  double get_gradient_norm_sq() const override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    double sum = 0.0;
    for(const auto& branch : _branches)
    {
      for(const auto& layer : branch.layers)
      {
        sum += layer->get_gradient_norm_sq();
      }
    }
    return sum;
  }

  inline void scale_temperature(double factor) noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto& branch : _branches)
    {
      for (auto& layer : branch.layers)
      {
        layer->scale_temperature(factor);
      }
    }
  }

  [[nodiscard]] inline double get_temperature(unsigned range_index) const noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    std::lock_guard<std::mutex> lock(_mutex);
#if VALIDATE_DATA == 1
    if (range_index >= _branches.size())
    {
      Logger::panic("Trying to get temperature for branch ", range_index, " which is out of bounds!");
    }
#endif
    // The temperature for a branch is in its last layer (the output layer)
    // and each branch head has exactly one range (range 0).
    return _branches[range_index].layers.back()->get_temperature(0);
  }

  inline void scale_temperature(unsigned range_index, double factor) noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    std::lock_guard<std::mutex> lock(_mutex);
#if VALIDATE_DATA == 1
    if (range_index >= _branches.size())
    {
      Logger::panic("Trying to scale temperature for branch ", range_index, " which is out of bounds!");
    }
#endif
    // Propagate scaling to the output layer of the specific branch
    _branches[range_index].layers.back()->scale_temperature(0, factor);
  }

  [[nodiscard]] const std::vector<Branch>& get_branches() const
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    return _branches;
  }
  [[nodiscard]] std::vector<Branch>& get_mutable_branches()
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    return _branches;
  }

  [[nodiscard]] std::vector<std::vector<NeuralNetworkHelperMetrics>> calculate_output_metrics(
    const std::vector<ErrorCalculation::type>& error_types,
    const std::vector<std::vector<double>>& checking_outputs,
    const std::vector<std::vector<double>>& predictions
  ) const override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    std::vector<std::vector<NeuralNetworkHelperMetrics>> errors;
    errors.reserve(number_output_layers());

    const size_t batch_size = predictions.size();
#if VALIDATE_DATA == 1
    if (batch_size != checking_outputs.size())
    {
      Logger::panic("The number of predictions is not the same as the number of given outputs!");
    }
#endif

    // Multi-output path logic works for 1 or more outputs
    std::vector<std::vector<double>> sliced_predictions(batch_size);
    std::vector<std::vector<double>> sliced_checking_outputs(batch_size);

    for (unsigned output_layer_index = 0; output_layer_index < number_output_layers(); ++output_layer_index)
    {
      std::vector<NeuralNetworkHelperMetrics> layer_errors;
      layer_errors.reserve(error_types.size());

      const auto& activation = output_layer_details()[output_layer_index].get_activation();
      const auto& activation_method = activation.get_method();

      const auto& bounds = layer_bounds(output_layer_index);
      const auto& configs = evaluation_config(output_layer_index);
      const size_t num_neurons = bounds.end - bounds.start + 1;

      for (size_t b = 0; b < batch_size; ++b)
      {
        sliced_predictions[b].assign(predictions[b].begin() + bounds.start, predictions[b].begin() + bounds.start + num_neurons);
        sliced_checking_outputs[b].assign(checking_outputs[b].begin() + bounds.start, checking_outputs[b].begin() + bounds.start + num_neurons);
      }

      for (const auto& error_type : error_types)
      {
        layer_errors.emplace_back(
          ErrorCalculation::calculate_error(error_type, sliced_checking_outputs, sliced_predictions, configs, activation_method),
          error_type);
      }
      errors.emplace_back(std::move(layer_errors));
    }
    return errors;
  }

private:
  static std::vector<OutputLayerDetails> extract_output_details(const std::vector<MultiOutputLayerDetails>& multi_output_layer_details)
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    std::vector<OutputLayerDetails> details;
    for (const auto& multi_output_layer_detail : multi_output_layer_details)
    {
      details.push_back(multi_output_layer_detail.get_output_details());
    }
    return details;
  }

  std::vector<Branch> _branches;
  mutable std::mutex _mutex;
};
