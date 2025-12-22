#pragma once

#include <vector>

struct LayerGradients {
  // For FFLayer weights or ElmanRNNLayer input-to-hidden
  std::vector<double> weights;
  // For biases
  std::vector<double> biases;
  // For ElmanRNNLayer recurrent weights
  std::vector<double> recurrent_weights;
  // For residual weights
  std::vector<double> residual_weights;
};
