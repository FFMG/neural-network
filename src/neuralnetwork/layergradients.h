#pragma once

#include <vector>

struct LayerGradients {
  // For FFLayer weights or ElmanRNNLayer input-to-hidden
  std::vector<std::vector<double>> weights;
  // For biases
  std::vector<double> biases;
  // For ElmanRNNLayer recurrent weights
  std::vector<std::vector<double>> recurrent_weights;
  // For residual weights
  std::vector<std::vector<double>> residual_weights;
};
