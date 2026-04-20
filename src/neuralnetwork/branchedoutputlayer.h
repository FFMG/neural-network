#pragma once

#include "layer.h"
#include "outputlayer.h"
#include "fflayer.h"
#include "ffoutputlayer.h"
#include "elmanrnnlayer.h"
#include "grurnnlayer.h"
#include <vector>
#include <memory>
#include <numeric>

class BranchedOutputLayer final : public Layer, public OutputLayer
{
public:
  struct Branch {
    std::vector<std::unique_ptr<Layer>> layers;
    // Internal buffers for this branch
    std::vector<GradientsAndOutputs> gradients_and_outputs;
    std::vector<HiddenStates> hidden_states;
    std::vector<unsigned> topology;
    
    Branch() = default;
    Branch(const Branch& src) {
      for (const auto& l : src.layers) layers.emplace_back(l->clone());
      topology = src.topology;
    }
    Branch(Branch&& src) noexcept : 
      layers(std::move(src.layers)),
      gradients_and_outputs(std::move(src.gradients_and_outputs)),
      hidden_states(std::move(src.hidden_states)),
      topology(std::move(src.topology))
    {}
    Branch& operator=(const Branch& src) {
      if (this != &src) {
        layers.clear();
        for (const auto& l : src.layers) layers.emplace_back(l->clone());
        topology = src.topology;
      }
      return *this;
    }
    Branch& operator=(Branch&& src) noexcept {
      if (this != &src) {
        layers = std::move(src.layers);
        gradients_and_outputs = std::move(src.gradients_and_outputs);
        hidden_states = std::move(src.hidden_states);
        topology = std::move(src.topology);
      }
      return *this;
    }

    void init_buffers(size_t batch_size) {
      if (gradients_and_outputs.size() < batch_size) {
        gradients_and_outputs.resize(batch_size, GradientsAndOutputs(topology));
        hidden_states.resize(batch_size, HiddenStates(topology));
      }
      for(size_t i=0; i<batch_size; ++i) {
        gradients_and_outputs[i].zero();
        hidden_states[i].zero();
      }
    }
  };

  BranchedOutputLayer(
    unsigned layer_index,
    unsigned num_inputs,
    unsigned num_outputs,
    const std::vector<LayerDetails::BranchDetails>& branched_details,
    int number_of_threads,
    bool has_bias
  ) :
    Layer(layer_index, Layer::LayerType::Branched, layer_activation_helper(activation(activation::method::linear, 0.0), num_inputs, num_outputs), OptimiserType::None, -1, {}, has_bias, {}, nullptr, number_of_threads, 0.0),
    OutputLayer(extract_output_details(branched_details))
  {
    MYODDWEB_PROFILE_FUNCTION("BranchedOutputLayer");
    
    for (const auto& bd : branched_details) {
      Branch branch;
      unsigned prev_n = num_inputs;
      branch.topology.push_back(num_inputs); // index 0 is input to branch
      
      // Hidden layers in branch
      for (size_t i = 0; i < bd.hidden_layers.size(); ++i) {
        const auto& ld = bd.hidden_layers[i];
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
      std::vector<OutputLayerDetails> olds = { bd.output_details };
      auto ol = std::make_unique<FFOutputLayer>(
        (unsigned)branch.layers.size() + 1,
        olds,
        prev_n,
        bd.output_details.get_size(),
        number_of_threads,
        has_bias
      );
      branch.layers.emplace_back(std::move(ol));
      branch.topology.push_back(bd.output_details.get_size());
      _branches.emplace_back(std::move(branch));
    }
  }

  BranchedOutputLayer(const BranchedOutputLayer& src) :
    Layer(src),
    OutputLayer(src),
    _branches(src._branches)
  {
    MYODDWEB_PROFILE_FUNCTION("BranchedOutputLayer");
  }

  BranchedOutputLayer(BranchedOutputLayer&& src) noexcept :
    Layer(std::move(src)),
    OutputLayer(std::move(src)),
    _branches(std::move(src._branches))
  {
    MYODDWEB_PROFILE_FUNCTION("BranchedOutputLayer");
  }

  virtual ~BranchedOutputLayer() = default;

  Layer* clone() const override { return new BranchedOutputLayer(*this); }

  void calculate_forward_feed(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& previous_layer,
    const std::vector<std::vector<double>>& /*batch_residual_output_values*/,
    std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    bool is_training) const override
  {
    MYODDWEB_PROFILE_FUNCTION("BranchedOutputLayer");
    
    const unsigned trunk_layer_index = previous_layer.get_layer_index();
    const unsigned this_layer_index = get_layer_index();

    for (auto& branch : const_cast<std::vector<Branch>&>(_branches)) {
      branch.init_buffers(batch_size);
    }

    for (size_t b = 0; b < batch_size; ++b) {
       std::vector<double> concatenated_output;
       concatenated_output.reserve(get_number_neurons());

       const auto trunk_output_span = batch_gradients_and_outputs[b].get_outputs(trunk_layer_index);
       std::vector<double> trunk_output(trunk_output_span.begin(), trunk_output_span.end());

       for (size_t i = 0; i < _branches.size(); ++i) {
         auto& branch = const_cast<Branch&>(_branches[i]);
         
         // The input to the branch is the trunk output
         branch.gradients_and_outputs[b].set_outputs(0, trunk_output);

         // Forward through branch layers
         for (size_t l_idx = 0; l_idx < branch.layers.size(); ++l_idx) {
           const auto& current_l = *branch.layers[l_idx];
           const Layer& prev_l = (l_idx == 0) ? 
             static_cast<const Layer&>(*this) : // We act as a proxy for input size
             *branch.layers[l_idx-1];

           // Note: BranchedOutputLayer needs to pass its own info as previous layer 
           // for the first layer of each branch to satisfy topology checks.
           
           // We'll use a slightly more direct approach since they are internal
           current_l.calculate_forward_feed(
             branch.gradients_and_outputs,
             prev_l,
             {}, // No residuals in branches for now
             branch.hidden_states,
             batch_size,
             is_training
           );
         }
         
         const auto b_out_span = branch.gradients_and_outputs[b].output_back();
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
     MYODDWEB_PROFILE_FUNCTION("BranchedOutputLayer");
     unsigned offset = 0;
     for (size_t i = 0; i < _branches.size(); ++i) {
       auto& branch = const_cast<Branch&>(_branches[i]);
       const unsigned b_out_size = branch.layers.back()->get_number_neurons();
       
       std::vector<std::vector<double>> sub_targets(batch_size);
       for(size_t b=0; b<batch_size; ++b) {
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
  void backprop_branches(size_t batch_size, int bptt_max_ticks) const {
    MYODDWEB_PROFILE_FUNCTION("BranchedOutputLayer");
    for (auto& branch : const_cast<std::vector<Branch>&>(_branches)) {
      for (int l_idx = (int)branch.layers.size() - 2; l_idx >= 0; --l_idx) {
        auto& current = *branch.layers[l_idx];
        const auto& next = *branch.layers[l_idx+1];
        
        std::vector<std::vector<double>> batch_next_gradients;
        batch_next_gradients.reserve(batch_size);
        for(size_t b=0; b<batch_size; ++b) {
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

  std::vector<std::vector<double>> get_trunk_gradients(size_t batch_size) const {
    MYODDWEB_PROFILE_FUNCTION("BranchedOutputLayer");
    std::vector<std::vector<double>> trunk_grads(batch_size, std::vector<double>(get_number_input_neurons(), 0.0));
    
    for (const auto& branch : _branches) {
      const auto& first_layer = *branch.layers.front();
      // To get gradients flowing back to trunk, we need: delta_first * W_first^T
      // This is exactly what calculate_hidden_gradients does for its input.
      
      // Since first_layer.calculate_hidden_gradients was called, 
      // the gradients at index 0 of branch.gradients_and_outputs[b] 
      // contain the gradients for the branch input (the trunk).
      
      for(size_t b=0; b<batch_size; ++b) {
        const auto g_span = branch.gradients_and_outputs[b].get_gradients(0);
        for(size_t j=0; j<trunk_grads[b].size(); ++j) {
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
    // This is handled by backprop_branches + get_trunk_gradients in Layers.cpp
  }

  void calculate_and_store_gradients(
    const std::vector<GradientsAndOutputs>& /*batch_gradients_and_outputs*/,
    const std::vector<HiddenStates>& /*hidden_states*/,
    const Layer& previous_layer,
    size_t batch_size,
    int bptt_max_ticks) override
  {
     MYODDWEB_PROFILE_FUNCTION("BranchedOutputLayer");
     for(auto& branch : _branches) {
       for(size_t l_idx = 0; l_idx < branch.layers.size(); ++l_idx) {
         auto& current_l = *branch.layers[l_idx];
         const Layer& prev_l = (l_idx == 0) ? previous_layer : *branch.layers[l_idx-1];
         
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
     for(auto& branch : _branches) {
       for(auto& layer : branch.layers) {
         layer->apply_stored_gradients(learning_rate, clipping_scale);
       }
     }
  }

  double get_gradient_norm_sq() const override
  {
    double sum = 0.0;
    for(const auto& branch : _branches) {
      for(const auto& layer : branch.layers) sum += layer->get_gradient_norm_sq();
    }
    return sum;
  }

private:
  static std::vector<OutputLayerDetails> extract_output_details(const std::vector<LayerDetails::BranchDetails>& branched_details) {
    std::vector<OutputLayerDetails> details;
    for (const auto& bd : branched_details) details.push_back(bd.output_details);
    return details;
  }

  std::vector<Branch> _branches;
};
