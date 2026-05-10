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
    Layer(0, Role::Input, activation(activation::method::linear, 0.0), OptimiserType::None, -1, num_neurons, num_neurons, {}, false, 0.0, nullptr, 1, 0.0)
  {
    MYODDWEB_PROFILE_FUNCTION("MultiInputProxyLayer");
  }

  [[nodiscard]] inline virtual Architecture get_layer_architecture() const override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiInputProxyLayer");
    return Architecture::MultiOutput;
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
  void calculate_hidden_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& next_layer,
    const std::vector<std::vector<double>>& batch_next_grad_matrix,
    const std::vector<HiddenStates>& /*batch_hidden_states*/,
    size_t batch_size,
    int /*bptt_max_ticks*/) const override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiInputProxyLayer");
    const size_t N_this = get_number_neurons();
    const size_t N_next = next_layer.get_number_neurons();

    for (size_t b = 0; b < batch_size; ++b)
    {
      const auto& next_grads = batch_next_grad_matrix[b];
      const size_t num_time_steps = next_grads.empty() ? 0 : next_grads.size() / N_next;
      
      std::vector<double> gradients_seq(num_time_steps * N_this, 0.0);
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        const double* g_t = &next_grads[t * N_next];
        double* dest_t = &gradients_seq[t * N_this];
        for (size_t j = 0; j < N_this; ++j)
        {
          double sum = 0.0;
          for (size_t k = 0; k < N_next; ++k)
          {
            sum += g_t[k] * next_layer.get_weight_value((unsigned)j, (unsigned)k);
          }
          dest_t[j] = sum;
        }
      }

      batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), gradients_seq);
      
      // Also provide a summed version in standard gradients for non-recurrent trunks
      std::vector<double> sum_grads(N_this, 0.0);
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        for (size_t j = 0; j < N_this; ++j)
        {
          sum_grads[j] += gradients_seq[t * N_this + j];
        }
      }
      batch_gradients_and_outputs[b].set_gradients(get_layer_index(), sum_grads);
    }
  }

  void calculate_hidden_gradients_from_output_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<std::vector<double>>& batch_output_gradients,
    const std::vector<HiddenStates>& /*batch_hidden_states*/,
    size_t batch_size,
    int /*bptt_max_ticks*/) const override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiInputProxyLayer");
    for (size_t b = 0; b < batch_size; ++b)
    {
      batch_gradients_and_outputs[b].set_gradients(get_layer_index(), batch_output_gradients[b]);
    }
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
    Layer(layer_index, Role::MultiOutput, create_layer_activation_helper(num_inputs, num_outputs, extract_output_details(multi_output_layer_details)), OptimiserType::None, -1, {}, has_bias, Layer::create_w_decays(num_inputs, num_outputs, 0.0), nullptr, number_of_threads, 0.0),
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
        auto l = Layer::create_hidden_layer(
          (unsigned)branch.layers.size() + 1, // index in branch starts at 1
          prev_n,
          ld,
          number_of_threads,
          has_bias
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

  [[nodiscard]] inline virtual Architecture get_layer_architecture() const override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    return Architecture::MultiOutput;
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
    size_t num_time_steps = 1;
    for (size_t b = 0; b < batch_size; ++b)
    {
       const auto trunk_rnn_output_span = batch_gradients_and_outputs[b].get_rnn_outputs(trunk_layer_index);
       const auto trunk_output_span = batch_gradients_and_outputs[b].get_outputs(trunk_layer_index);
       
       if (b == 0 && !trunk_rnn_output_span.empty())
       {
         num_time_steps = trunk_rnn_output_span.size() / get_number_input_neurons();
       }

       if (b == 0)
       {
         Logger::trace(
           [&trunk_layer_index, &trunk_output_span, &num_time_steps]
           {
             return Logger::factory("MultiOutputLayer [b=0] trunk_layer_index=", trunk_layer_index, " span_size=", trunk_output_span.size(), " ticks=", num_time_steps);
           });
       }

       std::vector<double> trunk_output(trunk_output_span.begin(), trunk_output_span.end());
       for (size_t i = 0; i < _branches.size(); ++i)
       {
         auto& branch = const_cast<Branch&>(_branches[i]);
         if (b == 0) 
         {
           Logger::trace(
             [&i, &branch] 
             {
               return Logger::factory("  Branch ", i, " input topology[0]=", branch.topology[0]);
             });
         }
         if (!trunk_rnn_output_span.empty())
         {
           branch.gradients_and_outputs[b].set_rnn_outputs(0, std::vector<double>(trunk_rnn_output_span.begin(), trunk_rnn_output_span.end()));
         }
         branch.gradients_and_outputs[b].set_outputs(0, trunk_output);
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
          const auto& prev_rnn_out = branch.gradients_and_outputs[b].get_rnn_outputs(prev_l.get_layer_index());
          const auto prev_std_out = branch.gradients_and_outputs[b].get_outputs(prev_l.get_layer_index());
          const size_t seq_size = !prev_rnn_out.empty() ? prev_rnn_out.size() : prev_std_out.size();
          const size_t n_prev = prev_l.get_number_neurons();
          const size_t l_num_time_steps = n_prev > 0 ? seq_size / n_prev : 0;
          branch.hidden_states[b].assign(current_l.get_layer_index(), l_num_time_steps, HiddenState(), current_l.get_pre_activation_multiplier());
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
       std::vector<double> concatenated_output_seq;
       concatenated_output_seq.reserve(num_time_steps * get_number_neurons());

       for (size_t t = 0; t < num_time_steps; ++t)
       {
         for (size_t i = 0; i < _branches.size(); ++i)
         {
           const auto& branch = _branches[i];
           const auto& b_rnn_out = branch.gradients_and_outputs[b].get_rnn_outputs(branch.layers.back()->get_layer_index());
           if (!b_rnn_out.empty())
           {
             const size_t b_out_n = branch.layers.back()->get_number_neurons();
             concatenated_output_seq.insert(concatenated_output_seq.end(), b_rnn_out.begin() + t * b_out_n, b_rnn_out.begin() + (t + 1) * b_out_n);
           }
           else
           {
             const auto b_out_span = branch.gradients_and_outputs[b].get_outputs(branch.layers.back()->get_layer_index());
             concatenated_output_seq.insert(concatenated_output_seq.end(), b_out_span.begin(), b_out_span.end());
           }
         }
       }

       batch_gradients_and_outputs[b].set_rnn_outputs(this_layer_index, concatenated_output_seq);
       
       std::vector<double> last_step_output(get_number_neurons());
       std::copy(concatenated_output_seq.end() - get_number_neurons(), concatenated_output_seq.end(), last_step_output.begin());
       batch_gradients_and_outputs[b].set_outputs(this_layer_index, last_step_output);

       if (!batch_hidden_states.empty())
       {
         // We must concatenate pre-activations for each time step if we want full BPTT support in following layers
         // but since this is usually the output layer, we might only need the views to be correctly sized.
         if (batch_hidden_states[b].at(this_layer_index).size() != num_time_steps)
         {
           batch_hidden_states[b].assign(this_layer_index, num_time_steps, HiddenState());
         }
         
         for (size_t t = 0; t < num_time_steps; ++t)
         {
           std::vector<double> concatenated_pre_act;
           concatenated_pre_act.reserve(get_number_neurons());
           for (size_t i = 0; i < _branches.size(); ++i)
           {
             const auto& branch = _branches[i];
             const auto branch_out_layer_idx = branch.layers.back()->get_layer_index();
             const auto b_pre_act = branch.hidden_states[b].at(branch_out_layer_idx)[t].get_pre_activation_sums();
             concatenated_pre_act.insert(concatenated_pre_act.end(), b_pre_act.begin(), b_pre_act.end());
           }
           batch_hidden_states[b].at(this_layer_index)[t].set_pre_activation_sums(concatenated_pre_act);
           // Hidden state values are already set via set_outputs/set_rnn_outputs above for the last step,
           // but we should set them for all steps in the hidden state object too.
           std::vector<double> step_output(get_number_neurons());
           std::copy(concatenated_output_seq.begin() + t * get_number_neurons(), concatenated_output_seq.begin() + (t+1) * get_number_neurons(), step_output.begin());
           batch_hidden_states[b].at(this_layer_index)[t].set_hidden_state_values(step_output);
         }
       }
    }
  }

  void calculate_output_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    std::vector<std::vector<double>>::const_iterator target_outputs_begin,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size) const override
  {
     MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
     (void)batch_gradients_and_outputs;
     (void)batch_hidden_states;
     (void)batch_size;
     std::lock_guard<std::mutex> lock(_mutex);
     unsigned offset = 0;
     const unsigned total_outputs = get_number_neurons();

     for (size_t i = 0; i < _branches.size(); ++i)
     {
       auto& branch = const_cast<Branch&>(_branches[i]);
       const auto& last_layer = *branch.layers.back();
       const unsigned b_out_size = last_layer.get_number_neurons();
       
       // Determine number of time steps from hidden states
       const size_t num_time_steps = branch.hidden_states[0].at(last_layer.get_layer_index()).size();

       std::vector<std::vector<double>> sub_targets(batch_size);
       for(size_t b=0; b<batch_size; ++b)
       {
         const auto& full_target = *(target_outputs_begin + b);
         if (full_target.size() == num_time_steps * total_outputs)
         {
           sub_targets[b].reserve(num_time_steps * b_out_size);
           for (size_t t = 0; t < num_time_steps; ++t)
           {
             sub_targets[b].insert(sub_targets[b].end(), 
               full_target.begin() + t * total_outputs + offset, 
               full_target.begin() + t * total_outputs + offset + b_out_size);
           }
         }
         else
         {
           // Fallback for single step target (or mismatched size)
           sub_targets[b].assign(full_target.begin() + offset, full_target.begin() + offset + b_out_size);
         }
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
    MultiInputProxyLayer proxy(get_number_input_neurons());

    for (auto& branch : const_cast<std::vector<Branch>&>(_branches))
    {
      // 1. Backprop through the branch layers (from Output to first Hidden)
      for (int l_idx = (int)branch.layers.size() - 1; l_idx >= 0; --l_idx)
      {
        auto& current = *branch.layers[l_idx];
        const auto& next = (l_idx == (int)branch.layers.size() - 1) ? current : *branch.layers[l_idx+1];
        
        // Skip output layer itself as its gradients were already set by calculate_output_gradients
        if (l_idx == (int)branch.layers.size() - 1) continue;

        std::vector<std::vector<double>> batch_next_gradients;
        batch_next_gradients.reserve(batch_size);
        for(size_t b=0; b<batch_size; ++b)
        {
          const auto rnn_span = branch.gradients_and_outputs[b].get_rnn_gradients(next.get_layer_index());
          if (!rnn_span.empty())
          {
            batch_next_gradients.emplace_back(rnn_span.begin(), rnn_span.end());
          }
          else
          {
            const auto g_span = branch.gradients_and_outputs[b].get_outputs(next.get_layer_index()); // Wait, this should be gradients
            // Actually, get_gradients is the last step.
            const auto std_g_span = branch.gradients_and_outputs[b].get_gradients(next.get_layer_index());
            batch_next_gradients.emplace_back(std_g_span.begin(), std_g_span.end());
          }
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

      // 2. Finally, calculate gradients for the branch input (index 0) using the proxy layer
      // This is what get_trunk_gradients() will sum up.
      const auto& first_layer = *branch.layers[0];

      std::vector<std::vector<double>> batch_first_gradients;
      batch_first_gradients.reserve(batch_size);
      for (size_t b = 0; b < batch_size; ++b)
      {
        const auto rnn_span = branch.gradients_and_outputs[b].get_rnn_gradients(first_layer.get_layer_index());
        if (!rnn_span.empty())
        {
          batch_first_gradients.emplace_back(rnn_span.begin(), rnn_span.end());
        }
        else
        {
          const auto g_span = branch.gradients_and_outputs[b].get_gradients(first_layer.get_layer_index());
          batch_first_gradients.emplace_back(g_span.begin(), g_span.end());
        }
      }
      proxy.calculate_hidden_gradients(
        branch.gradients_and_outputs,
        first_layer,
        batch_first_gradients,
        branch.hidden_states,
        batch_size,
        bptt_max_ticks
      );
    }
  }

  std::vector<std::vector<double>> get_trunk_gradients(size_t batch_size) const
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    std::lock_guard<std::mutex> lock(_mutex);
    
    // Determine if any branch is providing a sequence
    size_t max_seq_len = 1;
    for (const auto& branch : _branches)
    {
      if (batch_size > 0)
      {
         const auto rnn_span = branch.gradients_and_outputs[0].get_rnn_gradients(0);
         if (!rnn_span.empty())
         {
           max_seq_len = std::max(max_seq_len, rnn_span.size() / get_number_input_neurons());
         }
      }
    }

    std::vector<std::vector<double>> trunk_grads(batch_size, std::vector<double>(max_seq_len * get_number_input_neurons(), 0.0));
    const size_t N_trunk = get_number_input_neurons();

    for (const auto& branch : _branches)
    {
      for(size_t b=0; b<batch_size; ++b)
      {
        const auto rnn_span = branch.gradients_and_outputs[b].get_rnn_gradients(0);
        if (!rnn_span.empty())
        {
          const size_t branch_seq_len = rnn_span.size() / N_trunk;
          // If sequence lengths match, add element-wise
          if (branch_seq_len == max_seq_len)
          {
            for(size_t i=0; i<rnn_span.size(); ++i) trunk_grads[b][i] += rnn_span[i];
          }
          else if (branch_seq_len == 1)
          {
            // Broadcast single gradient to all time steps
            for(size_t t=0; t<max_seq_len; ++t)
              for(size_t j=0; j<N_trunk; ++j) trunk_grads[b][t*N_trunk + j] += rnn_span[j];
          }
        }
        else
        {
          const auto g_span = branch.gradients_and_outputs[b].get_gradients(0);
          if (!g_span.empty())
          {
            // Broadcast single gradient to all time steps
            for(size_t t=0; t<max_seq_len; ++t)
              for(size_t j=0; j<N_trunk; ++j) trunk_grads[b][t*N_trunk + j] += g_span[j];
          }
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

  void calculate_hidden_gradients_from_output_gradients(
    std::vector<GradientsAndOutputs>& /*batch_gradients_and_outputs*/,
    const std::vector<std::vector<double>>& /*batch_output_gradients*/,
    const std::vector<HiddenStates>& /*batch_hidden_states*/,
    size_t /*batch_size*/,
    int /*bptt_max_ticks*/) const override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    // This is handled by backprop_branches + get_trunk_gradients in Layers.cpp
  }

  void calculate_and_store_gradients(
    const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& hidden_states,
    const Layer& previous_layer,
    size_t batch_size,
    int bptt_max_ticks) override
  {
     MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
     (void)batch_gradients_and_outputs;
     (void)hidden_states;
     (void)previous_layer;
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

  [[nodiscard]] inline double get_inference_temperature(unsigned range_index) const noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    std::lock_guard<std::mutex> lock(_mutex);
#if VALIDATE_DATA == 1
    if (range_index >= _branches.size())
    {
      Logger::panic("Trying to get inference temperature for branch ", range_index, " which is out of bounds!");
    }
#endif
    return _branches[range_index].layers.back()->get_inference_temperature(0);
  }

  inline void set_inference_temperature(unsigned range_index, double t) noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    std::lock_guard<std::mutex> lock(_mutex);
#if VALIDATE_DATA == 1
    if (range_index >= _branches.size())
    {
      Logger::panic("Trying to set inference temperature for branch ", range_index, " which is out of bounds!");
    }
#endif
    _branches[range_index].layers.back()->set_inference_temperature(0, t);
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

    const unsigned total_outputs = get_number_neurons();

    for (unsigned output_layer_index = 0; output_layer_index < number_output_layers(); ++output_layer_index)
    {
      std::vector<NeuralNetworkHelperMetrics> layer_errors;
      layer_errors.reserve(error_types.size());

      const auto& activation = output_layer_details()[output_layer_index].get_activation();
      const auto& activation_method = activation.get_method();

      const auto& bounds = layer_bounds(output_layer_index);
      const auto& configs = evaluation_config(output_layer_index);
      const size_t num_neurons = bounds.end - bounds.start + 1;

      // Unroll sequences: treat each time step of each batch item as an independent sample for metrics.
      // This ensures that ErrorCalculation (which works on samples) correctly handles Softmax max-indices, etc.
      std::vector<std::vector<double>> unrolled_predictions;
      std::vector<std::vector<double>> unrolled_checking_outputs;

      for (size_t b = 0; b < batch_size; ++b)
      {
        const size_t p_total = predictions[b].size();
        const size_t c_total = checking_outputs[b].size();

        if (total_outputs == 0)
        {
          continue;
        }

        const size_t p_steps = p_total / total_outputs;
        const size_t c_steps = c_total / total_outputs;
        const size_t num_steps = std::min(p_steps, c_steps);

        // Align at the end. For example, if c_steps=1 and p_steps=10, we take the last prediction step.
        const size_t p_offset = (p_steps > num_steps) ? (p_steps - num_steps) : 0;
        const size_t c_offset = (c_steps > num_steps) ? (c_steps - num_steps) : 0;

        for (size_t t = 0; t < num_steps; ++t)
        {
          std::vector<double> p_slice(num_neurons);
          std::vector<double> c_slice(num_neurons);

          const auto p_start = predictions[b].begin() + (t + p_offset) * total_outputs + bounds.start;
          std::copy(p_start, p_start + num_neurons, p_slice.begin());

          const auto c_start = checking_outputs[b].begin() + (t + c_offset) * total_outputs + bounds.start;
          std::copy(c_start, c_start + num_neurons, c_slice.begin());

          unrolled_predictions.push_back(std::move(p_slice));
          unrolled_checking_outputs.push_back(std::move(c_slice));
        }
      }

      for (const auto& error_type : error_types)
      {
        layer_errors.emplace_back(
          ErrorCalculation::calculate_error(error_type, unrolled_checking_outputs, unrolled_predictions, configs, activation_method),
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

  static layer_activation_helper create_layer_activation_helper(unsigned num_inputs,
    unsigned num_neurons_in_this_layer,
    const std::vector<OutputLayerDetails>& output_layer_details)
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayer");
    layer_activation_helper lah(output_layer_details.front().get_activation(), num_inputs, num_neurons_in_this_layer);
    unsigned start = 0;
    unsigned end = 0;
    for (const auto& detail : output_layer_details)
    {
      end = start + detail.get_size();
      lah.set_bounds(detail.get_activation(), start, end);
      start = end;
    }
    return lah;
  }

  std::vector<Branch> _branches;
  mutable std::mutex _mutex;
};