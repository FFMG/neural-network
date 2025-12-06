#pragma once

#include <vector>
#include <cassert>

/**
 * @class HiddenState
 * @brief A simple container for the state of an entire layer at a single point in time.
 *
 * This class stores the pre-activation sums (z-values) and the final
 * activated output values (hidden state) for all neurons in a layer. It is
 * used to pass information between the forward and backward passes.
 */
class HiddenState
{
public:
    /**
     * @brief Default constructor.
     */
    HiddenState() = default;

    /**
     * @brief Constructor to initialize with a specific size.
     * @param num_neurons The number of neurons in the layer.
     */
    HiddenState(unsigned num_neurons)
      : _pre_activation_sums(num_neurons, 0.0),
        _hidden_state_values(num_neurons, 0.0)
    {
    }

    /**
     * @brief Sets the pre-activation sums for the entire layer.
     * @param sums A vector containing the pre-activation sum for each neuron.
     */
    void set_pre_activation_sums(const std::vector<double>& sums)
    {
        _pre_activation_sums = sums;
    }
    
    /**
     * @brief Sets the activated output values (hidden state) for the entire layer.
     * @param values A vector containing the activated output for each neuron.
     */
    void set_hidden_state_values(const std::vector<double>& values)
    {
        _hidden_state_values = values;
    }

    /**
     * @brief Gets the pre-activation sum for a specific neuron.
     * @param neuron_index The index of the neuron.
     * @return The pre-activation sum (z-value).
     */
    double get_pre_activation_sum_at_neuron(unsigned neuron_index) const
    {
        assert(neuron_index < _pre_activation_sums.size());
        return _pre_activation_sums[neuron_index];
    }
    
    /**
     * @brief Gets the entire vector of pre-activation sums.
     * @return A const reference to the vector of sums.
     */
    const std::vector<double>& get_pre_activation_sums() const
    {
        return _pre_activation_sums;
    }

    /**
     * @brief Gets the activated output value (hidden state) for a specific neuron.
     * @param neuron_index The index of the neuron.
     * @return The activated output value.
     */
    double get_hidden_state_value_at_neuron(unsigned neuron_index) const
    {
        assert(neuron_index < _hidden_state_values.size());
        return _hidden_state_values[neuron_index];
    }

    /**
     * @brief Gets the entire vector of hidden state values.
     * @return A const reference to the vector of activated outputs.
     */
    const std::vector<double>& get_hidden_state_values() const 
    {
        return _hidden_state_values;
    }

private:
    std::vector<double> _pre_activation_sums;
    std::vector<double> _hidden_state_values;
};