#include "common/activation.h"
#include "helpers/errorcalculation.h"
#include "neuralnetwork.h"
#include "neuralnetworkoptions.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>


using namespace myoddweb::nn;
namespace math_expect {
  // --- Basic Formulas ---

  double mae(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred) {
    double total = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      for (size_t j = 0; j < gt[i].size(); ++j) {
        total += std::abs(gt[i][j] - pred[i][j]);
        ++count;
      }
    }
    return count == 0 ? 0.0 : total / count;
  }

  double mse(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred) {
    double total = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      for (size_t j = 0; j < gt[i].size(); ++j) {
        double diff = gt[i][j] - pred[i][j];
        total += diff * diff;
        ++count;
      }
    }
    return count == 0 ? 0.0 : total / count;
  }

  double rmse(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred) {
    double total_rmse = 0.0;
    size_t seq_count = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      double seq_mse = 0.0;
      for (size_t j = 0; j < gt[i].size(); ++j) {
        double diff = gt[i][j] - pred[i][j];
        seq_mse += diff * diff;
      }
      total_rmse += std::sqrt(seq_mse / gt[i].size());
      ++seq_count;
    }
    return seq_count == 0 ? 0.0 : total_rmse / seq_count;
  }

  double nrmse(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred, double eps = 1e-12) {
    double total_nrmse = 0.0;
    size_t seq_count = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      double seq_mse = 0.0;
      double min_val = gt[i][0], max_val = gt[i][0], mean_abs = 0.0;
      for (size_t j = 0; j < gt[i].size(); ++j) {
        double diff = gt[i][j] - pred[i][j];
        seq_mse += diff * diff;
        min_val = std::min(min_val, gt[i][j]);
        max_val = std::max(max_val, gt[i][j]);
        mean_abs += std::abs(gt[i][j]);
      }
      double rmse_val = std::sqrt(seq_mse / gt[i].size());
      double denom = max_val - min_val;
      if (denom < eps) denom = mean_abs / gt[i].size();
      if (denom >= eps) {
        total_nrmse += rmse_val / denom;
        ++seq_count;
      }
    }
    return seq_count == 0 ? 0.0 : total_nrmse / seq_count;
  }

  double huber(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred, double delta) {
    double total = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      for (size_t j = 0; j < gt[i].size(); ++j) {
        double a = std::abs(gt[i][j] - pred[i][j]);
        if (a <= delta) {
          total += 0.5 * a * a;
        } else {
          total += delta * (a - 0.5 * delta);
        }
        ++count;
      }
    }
    return count == 0 ? 0.0 : total / count;
  }

  double huber_direction(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred, double delta, double lambda) {
    double total = 0.0;
    size_t count = 0;
    const double scale = 100.0;
    for (size_t i = 0; i < gt.size(); ++i) {
      for (size_t j = 0; j < gt[i].size(); ++j) {
        double t = gt[i][j];
        double o = pred[i][j];
        double a = std::abs(t - o);
        double loss = (a <= delta) ? (0.5 * a * a) : (delta * (a - 0.5 * delta));
        
        if (std::abs(t) > 1e-6) {
          double x = -scale * t * o;
          double dir_loss = (x > 0.0) ? (x + std::log1p(std::exp(-x))) : std::log1p(std::exp(x));
          loss += lambda * dir_loss;
        }
        total += loss;
        ++count;
      }
    }
    return count == 0 ? 0.0 : total / count;
  }

  double log_cosh(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred) {
    double total = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      for (size_t j = 0; j < gt[i].size(); ++j) {
        double x = std::abs(gt[i][j] - pred[i][j]);
        total += x + std::log1p(std::exp(-2.0 * x)) - std::log(2.0);
        ++count;
      }
    }
    return count == 0 ? 0.0 : total / count;
  }

  double mape(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred, double epsilon) {
    double total_mape = 0.0;
    size_t seq_count = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      double seq_sum = 0.0;
      size_t count = 0;
      for (size_t j = 0; j < gt[i].size(); ++j) {
        double denom = std::abs(gt[i][j]);
        if (denom >= epsilon) {
          seq_sum += std::abs(gt[i][j] - pred[i][j]) / denom;
          ++count;
        }
      }
      if (count > 0) {
        total_mape += seq_sum / count;
        ++seq_count;
      }
    }
    return seq_count == 0 ? 0.0 : total_mape / seq_count;
  }

  double smape(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred, double epsilon) {
    double total_smape = 0.0;
    size_t seq_count = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      double seq_sum = 0.0;
      size_t count = 0;
      for (size_t j = 0; j < gt[i].size(); ++j) {
        double denom = (std::abs(gt[i][j]) + std::abs(pred[i][j])) / 2.0;
        if (denom >= epsilon) {
          seq_sum += std::abs(gt[i][j] - pred[i][j]) / denom;
          ++count;
        }
      }
      if (count > 0) {
        total_smape += seq_sum / count;
        ++seq_count;
      }
    }
    return seq_count == 0 ? 0.0 : total_smape / seq_count;
  }

  double wape(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred) {
    double abs_err = 0.0;
    double abs_actual = 0.0;
    for (size_t i = 0; i < gt.size(); ++i) {
      for (size_t j = 0; j < gt[i].size(); ++j) {
        abs_err += std::abs(gt[i][j] - pred[i][j]);
        abs_actual += std::abs(gt[i][j]);
      }
    }
    return abs_actual == 0.0 ? (abs_err == 0.0 ? 0.0 : 1.0) : abs_err / abs_actual;
  }

  double bce(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred, double epsilon) {
    double total = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      for (size_t j = 0; j < gt[i].size(); ++j) {
        double p = std::clamp(pred[i][j], epsilon, 1.0 - epsilon);
        double y = gt[i][j];
        total += -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
        ++count;
      }
    }
    return count == 0 ? 0.0 : total / count;
  }

  double cross_entropy(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred, double lambda) {
    const double eps = 1e-12;
    double total_loss = 0.0;
    size_t seq_count = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      double sample_loss = 0.0;
      for (size_t j = 0; j < gt[i].size(); ++j) {
        if (gt[i][j] > 0.0) {
          double p = std::clamp(pred[i][j], eps, 1.0 - eps);
          sample_loss += -gt[i][j] * std::log(p);
        }
      }
      total_loss += sample_loss;
      ++seq_count;
    }
    return seq_count == 0 ? 0.0 : (total_loss / seq_count) * lambda;
  }

  // --- Branching Logic ---

  double directional_accuracy(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred, double baseline, double neutral_tolerance) {
    size_t correct = 0;
    size_t total = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      for (size_t j = 0; j < gt[i].size(); ++j) {
        if (std::abs(gt[i][j] - baseline) < neutral_tolerance) continue;
        bool gt_up = gt[i][j] > baseline;
        bool pr_up = pred[i][j] > baseline;
        if (gt_up == pr_up) ++correct;
        ++total;
      }
    }
    return total == 0 ? 0.0 : static_cast<double>(correct) / total;
  }

  double softmax_directional_accuracy(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred) {
    size_t correct = 0;
    size_t total = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      const size_t num_classes = pred[i].size();
      const double mid = (static_cast<double>(num_classes) - 1.0) / 2.0;
      size_t pred_idx = std::distance(pred[i].begin(), std::max_element(pred[i].begin(), pred[i].end()));
      size_t gt_idx = std::distance(gt[i].begin(), std::max_element(gt[i].begin(), gt[i].end()));
      
      if (std::abs(static_cast<double>(gt_idx) - mid) < 0.1) continue;
      bool pred_up = static_cast<double>(pred_idx) > mid;
      bool gt_up = static_cast<double>(gt_idx) > mid;
      if (pred_up == gt_up && std::abs(static_cast<double>(pred_idx) - mid) >= 0.1) ++correct;
      ++total;
    }
    return total == 0 ? 0.0 : static_cast<double>(correct) / total;
  }

  double directional_confidence_score(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred, double baseline, double neutral_tolerance, double confidence_threshold) {
    size_t correct = 0;
    size_t total = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      for (size_t j = 0; j < gt[i].size(); ++j) {
        if (std::abs(gt[i][j] - baseline) < neutral_tolerance) continue;
        if (std::abs(pred[i][j] - baseline) < confidence_threshold) continue;
        bool gt_up = gt[i][j] > baseline;
        bool pr_up = pred[i][j] > baseline;
        if (gt_up == pr_up) ++correct;
        ++total;
      }
    }
    return total == 0 ? 0.0 : static_cast<double>(correct) / total;
  }

  double softmax_directional_confidence_score(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred, double confidence_threshold) {
    size_t correct = 0;
    size_t total = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      const size_t num_classes = pred[i].size();
      const double mid = (static_cast<double>(num_classes) - 1.0) / 2.0;
      auto max_pred_it = std::max_element(pred[i].begin(), pred[i].end());
      size_t pred_idx = std::distance(pred[i].begin(), max_pred_it);
      double confidence = *max_pred_it;
      size_t gt_idx = std::distance(gt[i].begin(), std::max_element(gt[i].begin(), gt[i].end()));

      if (std::abs(static_cast<double>(pred_idx) - mid) < 0.1 || confidence < confidence_threshold) continue;
      if (std::abs(static_cast<double>(gt_idx) - mid) < 0.1) continue;
      
      bool pred_up = static_cast<double>(pred_idx) > mid;
      bool gt_up = static_cast<double>(gt_idx) > mid;
      if (pred_up == gt_up) ++correct;
      ++total;
    }
    return total == 0 ? 0.0 : static_cast<double>(correct) / total;
  }

  double prediction_coverage(const std::vector<std::vector<double>>& pred, double confidence_threshold, double baseline = 0.0) {
    size_t covered = 0;
    for (const auto& seq : pred) {
      bool any = false;
      for (double v : seq) {
        if (std::abs(v - baseline) > confidence_threshold) { any = true; break; }
      }
      if (any) ++covered;
    }
    return pred.empty() ? 0.0 : static_cast<double>(covered) / pred.size();
  }

  double softmax_prediction_coverage(const std::vector<std::vector<double>>& pred, double confidence_threshold) {
    size_t covered = 0;
    for (const auto& seq : pred) {
      if (seq.empty()) continue;
      double max_v = *std::max_element(seq.begin(), seq.end());
      if (max_v > confidence_threshold) ++covered;
    }
    return pred.empty() ? 0.0 : static_cast<double>(covered) / pred.size();
  }
}

class ErrorCalculationTest : public ::testing::Test {
protected:
  // Complex data set
  std::vector<std::vector<double>> ground_truth = {
    {1.0, 0.5, -0.2, 0.0, 0.8},
    {0.0, 1.0, 0.8, -0.5, 0.2},
    {-1.0, -0.8, 0.0, 0.5, 1.0}
  };
  std::vector<std::vector<double>> predictions = {
    {0.9, 0.6, -0.1, 0.05, 0.7},
    {0.2, 0.8, 0.7, -0.4, 0.3},
    {-0.9, -0.7, 0.1, 0.4, 0.8}
  };

  EvaluationConfig config{0.05, 0.5, 1.0, 0.1, true, 1.0, 1e-7};
  double tolerance = 1e-9;

  void SetUp() override {}
};

// --- Basic Formula Verification ---

TEST_F(ErrorCalculationTest, MAE) {
  double expected = math_expect::mae(ground_truth, predictions);
  double actual = ErrorCalculation::calculate_mae_error(ground_truth, predictions);
  EXPECT_NEAR(expected, actual, tolerance);
}

TEST_F(ErrorCalculationTest, MSE) {
  double expected = math_expect::mse(ground_truth, predictions);
  double actual = ErrorCalculation::calculate_mse_error(ground_truth, predictions);
  EXPECT_NEAR(expected, actual, tolerance);
}

TEST_F(ErrorCalculationTest, RMSE) {
  double expected = math_expect::rmse(ground_truth, predictions);
  double actual = ErrorCalculation::calculate_rmse_error(ground_truth, predictions);
  EXPECT_NEAR(expected, actual, tolerance);
}

TEST_F(ErrorCalculationTest, NRMSE) {
  double expected = math_expect::nrmse(ground_truth, predictions, config.epsilon());
  double actual = ErrorCalculation::calculate_nrmse_error(ground_truth, predictions);
  EXPECT_NEAR(expected, actual, tolerance);
}

TEST_F(ErrorCalculationTest, Huber) {
  double expected = math_expect::huber(ground_truth, predictions, config.huber_delta());
  double actual = ErrorCalculation::calculate_huber_loss_error(ground_truth, predictions, config);
  EXPECT_NEAR(expected, actual, tolerance);
}

TEST_F(ErrorCalculationTest, HuberDirection) {
  double expected = math_expect::huber_direction(ground_truth, predictions, config.huber_delta(), config.direction_lambda());
  double actual = ErrorCalculation::calculate_huber_direction_loss(ground_truth, predictions, config);
  EXPECT_NEAR(expected, actual, 1e-8);
}

TEST_F(ErrorCalculationTest, LogCosh) {
  double expected = math_expect::log_cosh(ground_truth, predictions);
  double actual = ErrorCalculation::calculate_log_cosh(ground_truth, predictions);
  EXPECT_NEAR(expected, actual, tolerance);
}

TEST_F(ErrorCalculationTest, MAPE) {
  double expected = math_expect::mape(ground_truth, predictions, config.epsilon());
  double actual = ErrorCalculation::calculate_forecast_mape(ground_truth, predictions, config);
  EXPECT_NEAR(expected, actual, tolerance);
}

TEST_F(ErrorCalculationTest, sMAPE) {
  double expected = math_expect::smape(ground_truth, predictions, config.epsilon());
  double actual = ErrorCalculation::calculate_forecast_smape(ground_truth, predictions, config);
  EXPECT_NEAR(expected, actual, tolerance);
}

TEST_F(ErrorCalculationTest, WAPE) {
  double expected = math_expect::wape(ground_truth, predictions);
  double actual = ErrorCalculation::calculate_forecast_wape(ground_truth, predictions);
  EXPECT_NEAR(expected, actual, tolerance);
}

TEST_F(ErrorCalculationTest, BCE) {
  double expected = math_expect::bce(ground_truth, predictions, config.epsilon());
  double actual = ErrorCalculation::calculate_bce_loss(ground_truth, predictions, config);
  EXPECT_NEAR(expected, actual, tolerance);
}

TEST_F(ErrorCalculationTest, CrossEntropy) {
  double expected = math_expect::cross_entropy(ground_truth, predictions, config.cross_entropy_lambda());
  double actual = ErrorCalculation::calculate_cross_entropy(ground_truth, predictions, config);
  EXPECT_NEAR(expected, actual, tolerance);
}

// --- Branching Logic Verification ---

TEST_F(ErrorCalculationTest, DirectionalAccuracyLinear) {
  double expected = math_expect::directional_accuracy(ground_truth, predictions, 0.0, config.neutral_tolerance());
  double actual = ErrorCalculation::calculate_directional_accuracy(ground_truth, predictions, config, activation::method::linear);
  EXPECT_DOUBLE_EQ(expected, actual);
}

TEST_F(ErrorCalculationTest, DirectionalAccuracySigmoid) {
  double expected = math_expect::directional_accuracy(ground_truth, predictions, 0.5, config.neutral_tolerance());
  double actual = ErrorCalculation::calculate_directional_accuracy(ground_truth, predictions, config, activation::method::sigmoid);
  EXPECT_DOUBLE_EQ(expected, actual);
}

TEST_F(ErrorCalculationTest, SoftmaxDirectionalAccuracy) {
  // Use a specific softmax data set
  std::vector<std::vector<double>> gt_sm = {{1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}}; // DOWN, UP, NEUTRAL
  std::vector<std::vector<double>> pr_sm = {{0.8, 0.1, 0.1}, {0.1, 0.1, 0.8}, {0.3, 0.4, 0.3}}; // Match, Match, Neutral Match
  double expected = math_expect::softmax_directional_accuracy(gt_sm, pr_sm);
  double actual = ErrorCalculation::calculate_softmax_directional_accuracy(gt_sm, pr_sm);
  EXPECT_DOUBLE_EQ(expected, actual);
}

TEST_F(ErrorCalculationTest, DirectionalConfidenceScoreTanh) {
  double expected = math_expect::directional_confidence_score(ground_truth, predictions, 0.0, config.neutral_tolerance(), config.confidence_threshold());
  double actual = ErrorCalculation::calculate_directional_confidence_score(ground_truth, predictions, config, activation::method::tanh);
  EXPECT_DOUBLE_EQ(expected, actual);
}

TEST_F(ErrorCalculationTest, SoftmaxDirectionalConfidenceScore) {
  std::vector<std::vector<double>> gt_sm = {{1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}};
  std::vector<std::vector<double>> pr_sm = {{0.6, 0.2, 0.2}, {0.4, 0.3, 0.3}, {0.2, 0.6, 0.2}}; 
  // Seq 1: DOWN vs DOWN, Conf 0.6 > 0.5 -> Match
  // Seq 2: UP vs DOWN, Conf 0.4 <= 0.5 -> Ignore
  // Seq 3: NEUTRAL -> Ignore
  double expected = math_expect::softmax_directional_confidence_score(gt_sm, pr_sm, config.confidence_threshold());
  double actual = ErrorCalculation::calculate_softmax_directional_confidence_score(gt_sm, pr_sm, config);
  EXPECT_DOUBLE_EQ(expected, actual);
}

TEST_F(ErrorCalculationTest, PredictionCoverageRelu) {
  double expected = math_expect::prediction_coverage(predictions, config.confidence_threshold());
  double actual = ErrorCalculation::calculate_prediction_coverage(predictions, config, activation::method::relu);
  EXPECT_DOUBLE_EQ(expected, actual);
}

TEST_F(ErrorCalculationTest, SoftmaxPredictionCoverage) {
  double expected = math_expect::softmax_prediction_coverage(predictions, config.confidence_threshold());
  double actual = ErrorCalculation::calculate_softmax_prediction_coverage(predictions, config);
  EXPECT_DOUBLE_EQ(expected, actual);
}

TEST_F(ErrorCalculationTest, PredictionCoverageSigmoidUsesNeutralBaseline) {
  // Sigmoid's neutral point is 0.5, not 0.0: a value near 0.0 is a *confident* "down" call,
  // not an unconfident one. calculate_prediction_coverage must measure distance from 0.5,
  // exactly like calculate_directional_confidence_score already does for sigmoid.
  EvaluationConfig sigmoid_config{ 0.05, 0.3, 1.0, 0.1, true, 1.0, 1e-7 };
  std::vector<std::vector<double>> sigmoid_predictions = {
    { 0.05 },  // |0.05 - 0.5| = 0.45 > 0.3 -> confident (strongly "down")
    { 0.95 },  // |0.95 - 0.5| = 0.45 > 0.3 -> confident (strongly "up")
    { 0.5 },   // |0.5  - 0.5| = 0.0  <= 0.3 -> not confident (neutral)
    { 0.55 },  // |0.55 - 0.5| = 0.05 <= 0.3 -> not confident
  };

  double expected = math_expect::prediction_coverage(sigmoid_predictions, sigmoid_config.confidence_threshold(), 0.5);
  double actual = ErrorCalculation::calculate_prediction_coverage(sigmoid_predictions, sigmoid_config, activation::method::sigmoid);
  EXPECT_DOUBLE_EQ(expected, actual);
  EXPECT_DOUBLE_EQ(actual, 0.5); // 2 out of 4 sequences (0.05 and 0.95) are confident

  // The raw-magnitude (unfixed) interpretation would score 0.75 instead: 0.95, 0.5, and 0.55
  // all exceed the 0.3 threshold in raw magnitude, wrongly counting the neutral 0.5 value as
  // "confident" while still correctly catching 0.95 -- a different, wrong answer either way.
  double raw_magnitude_interpretation = math_expect::prediction_coverage(sigmoid_predictions, sigmoid_config.confidence_threshold(), 0.0);
  EXPECT_DOUBLE_EQ(raw_magnitude_interpretation, 0.75);
  EXPECT_NE(actual, raw_magnitude_interpretation);
}

TEST_F(ErrorCalculationTest, PredictionCoverageEmptySequencePanics) {
  std::vector<std::vector<double>> pred_with_empty = { { 0.9 }, {} };
  EXPECT_THROW((void)ErrorCalculation::calculate_prediction_coverage(pred_with_empty, config, activation::method::linear), std::runtime_error);
}

// --- Complex Multi-Layer Integration Test ---

TEST_F(ErrorCalculationTest, ComplexArchitectureIntegration) {
  // Input(2) -> Hidden1(FF, 3, ReLU) -> Hidden2(RNN, 3, Tanh) -> Output(FF, 1, Sigmoid)
  auto options = NeuralNetworkOptions::create({2, 3, 3, 1})
         .with_log_level(Logger::LogLevel::Information)
         .with_hidden_layers({
           LayerDetails(Layer::Architecture::FF, 3, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::None, 0.0, false, 0, 0, 0, 0, 0, 0, 0),
           LayerDetails(Layer::Architecture::Elman, 3, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::None, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
         })
         .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::sigmoid, 0.0), ErrorCalculation::type::mse, config, 0.0, OptimiserType::None, 0.0))
         .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = {{1.0, 2.0}, {0.5, -0.5}};
  std::vector<std::vector<double>> targets = {{0.8}, {0.1}};

  // 1. Get predictions via forward pass
  auto prediction = nn.think(inputs);
  ASSERT_EQ(prediction.size(), 2);

  // 2. Calculate error directly using ErrorCalculation
  // We unroll predictions and targets as ErrorCalculation expects spans of vectors
  std::vector<std::vector<double>> unrolled_pred = prediction;
  std::vector<std::vector<double>> unrolled_gt = targets;

  double actual_mse = ErrorCalculation::calculate_error(
    ErrorCalculation::type::mse, 
    unrolled_gt, 
    unrolled_pred, 
    config, 
    activation::method::sigmoid
  );

  // 3. Calculate expected error mathematically
  double expected_mse = math_expect::mse(targets, prediction);
  
  EXPECT_NEAR(actual_mse, expected_mse, 1e-12);
}

TEST_F(ErrorCalculationTest, MultiOutputArchitectureIntegration) {
  // Test with multiple outputs to ensure unrolling works
  auto options = NeuralNetworkOptions::create({2, 5, 3}) // 3 outputs
         .with_log_level(Logger::LogLevel::Information)
         .with_hidden_layers({
           LayerDetails(Layer::Architecture::FF, 5, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::None, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
         })
         .with_output_layer_details(OutputLayerDetails(3, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, config, 0.0, OptimiserType::None, 0.0))
         .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = {{1.0, 0.5}, {0.2, 0.8}};
  std::vector<std::vector<double>> targets = {{0.9, 0.1, 0.5}, {0.2, 0.7, 0.3}};

  auto prediction = nn.think(inputs);
  
  // Calculate errors for all types directly
  std::vector<ErrorCalculation::type> types = {
    ErrorCalculation::type::mae, 
    ErrorCalculation::type::mse, 
    ErrorCalculation::type::rmse,
    ErrorCalculation::type::nrmse,
    ErrorCalculation::type::mape,
    ErrorCalculation::type::smape,
    ErrorCalculation::type::wape,
    ErrorCalculation::type::huber_loss,
    ErrorCalculation::type::log_cosh
  };

  for(const auto& type : types) {
    double actual = ErrorCalculation::calculate_error(
      type, 
      targets, 
      prediction, 
      config, 
      activation::method::linear
    );

    double expected = 0.0;
    switch(type) {
      case ErrorCalculation::type::mae: expected = math_expect::mae(targets, prediction); break;
      case ErrorCalculation::type::mse: expected = math_expect::mse(targets, prediction); break;
      case ErrorCalculation::type::rmse: expected = math_expect::rmse(targets, prediction); break;
      case ErrorCalculation::type::nrmse: expected = math_expect::nrmse(targets, prediction); break;
      case ErrorCalculation::type::mape: expected = math_expect::mape(targets, prediction, config.epsilon()); break;
      case ErrorCalculation::type::smape: expected = math_expect::smape(targets, prediction, config.epsilon()); break;
      case ErrorCalculation::type::wape: expected = math_expect::wape(targets, prediction); break;
      case ErrorCalculation::type::huber_loss: expected = math_expect::huber(targets, prediction, config.huber_delta()); break;
      case ErrorCalculation::type::log_cosh: expected = math_expect::log_cosh(targets, prediction); break;
      default: break;
    }
    EXPECT_NEAR(actual, expected, 1e-7) << "Failed for type: " << ErrorCalculation::type_to_string(type);
  }
}



TEST_F(ErrorCalculationTest, EvaluationConfigSensitivity) {
  // Test how changing EvaluationConfig values affects the results
  std::vector<std::vector<double>> gt = {{1.0}};
  std::vector<std::vector<double>> pr = {{0.5}};

  // Huber delta = 0.1. Error = 0.5. Abs error = 0.5 > 0.1.
  // Loss = 0.1 * (0.5 - 0.5 * 0.1) = 0.1 * 0.45 = 0.045
  EvaluationConfig c1{0.0, 0.0, 0.1, 0.0, false, 0.0, 1e-12};
  double h1 = ErrorCalculation::calculate_huber_loss_error(gt, pr, c1);
  EXPECT_NEAR(h1, 0.045, tolerance);

  // Huber delta = 1.0. Error = 0.5. Abs error = 0.5 < 1.0.
  // Loss = 0.5 * 0.5^2 = 0.125
  EvaluationConfig c2{0.0, 0.0, 1.0, 0.0, false, 0.0, 1e-12};
  double h2 = ErrorCalculation::calculate_huber_loss_error(gt, pr, c2);
  EXPECT_NEAR(h2, 0.125, tolerance);
  
  // Direction Lambda sensitivity
  // Use a mismatch to ensure direction penalty is significant
  std::vector<std::vector<double>> pr_mismatch = {{-0.5}};
  
  // Huber Direction Loss = Huber + lambda * direction_loss
  // If lambda is 0, should be same as huber
  EvaluationConfig c3{0.0, 0.0, 1.0, 0.0, true, 0.0, 1e-12};
  double hd1 = ErrorCalculation::calculate_huber_direction_loss(gt, pr_mismatch, c3);
  // error = 1.0 - (-0.5) = 1.5. delta = 1.0. 
  // huber = 1.0 * (1.5 - 0.5 * 1.0) = 1.0.
  EXPECT_NEAR(hd1, 1.0, tolerance);

  EvaluationConfig c4{0.0, 0.0, 1.0, 1.0, true, 0.0, 1e-12};
  double hd2 = ErrorCalculation::calculate_huber_direction_loss(gt, pr_mismatch, c4);
  EXPECT_GT(hd2, hd1); // Should be higher due to direction penalty

  // Verify that use_direction_penalty flag is respected
  EvaluationConfig c5{0.0, 0.0, 1.0, 1.0, false, 0.0, 1e-12};
  double hd3 = ErrorCalculation::calculate_huber_direction_loss(gt, pr_mismatch, c5);
  EXPECT_DOUBLE_EQ(hd3, hd1); // Should ignore lambda because flag is false
}

TEST_F(ErrorCalculationTest, ActivationMethodBranching) {
  // Test all activation types to ensure they don't crash and follow linear/sigmoid/softmax logic
  std::vector<activation::method> methods = {
    activation::method::linear, activation::method::sigmoid, activation::method::tanh,
    activation::method::relu, activation::method::leakyRelu, activation::method::PRelu,
    activation::method::selu, activation::method::swish, activation::method::mish,
    activation::method::gelu, activation::method::elu, activation::method::softmax
  };

  for (auto m : methods) {
    if (m == activation::method::softmax) {
      EXPECT_NO_THROW((void)ErrorCalculation::calculate_error(ErrorCalculation::type::directional_accuracy, ground_truth, predictions, config, m));
    } else {
      EXPECT_NO_THROW((void)ErrorCalculation::calculate_error(ErrorCalculation::type::directional_accuracy, ground_truth, predictions, config, m));
    }
  }
}

TEST_F(ErrorCalculationTest, LogCoshCalculationAndStringTypeTest)
{
  std::vector<std::vector<double>> gt = { { 0.0, 1.0, -1.0 } };
  std::vector<std::vector<double>> pred = { { 0.5, 0.5, -0.5 } };

  double actual = ErrorCalculation::calculate_log_cosh(gt, pred);
  double expected = math_expect::log_cosh(gt, pred);
  EXPECT_NEAR(actual, expected, 1e-12);

  EXPECT_EQ(ErrorCalculation::string_to_type("Log-Cosh"), ErrorCalculation::type::log_cosh);
  EXPECT_EQ(ErrorCalculation::string_to_type("MSE"), ErrorCalculation::type::mse);
}

TEST_F(ErrorCalculationTest, AllTypesStringRoundtripCoverage)
{
  const std::vector<ErrorCalculation::type> all_types = {
    ErrorCalculation::type::none,
    ErrorCalculation::type::huber_loss,
    ErrorCalculation::type::huber_direction_loss,
    ErrorCalculation::type::mae,
    ErrorCalculation::type::mse,
    ErrorCalculation::type::rmse,
    ErrorCalculation::type::nrmse,
    ErrorCalculation::type::mape,
    ErrorCalculation::type::smape,
    ErrorCalculation::type::wape,
    ErrorCalculation::type::directional_accuracy,
    ErrorCalculation::type::bce_loss,
    ErrorCalculation::type::cross_entropy,
    ErrorCalculation::type::log_cosh,
    ErrorCalculation::type::directional_confidence_score,
    ErrorCalculation::type::prediction_coverage
  };

  for (const auto& t : all_types)
  {
    const std::string str = ErrorCalculation::type_to_string(t);
    EXPECT_FALSE(str.empty());
    const ErrorCalculation::type back = ErrorCalculation::string_to_type(str);
    EXPECT_EQ(t, back) << "Failed roundtrip for type: " << str;
  }
}

TEST_F(ErrorCalculationTest, HandCalculatedAnalyticalProofs)
{
  // 1. MAE & MSE on deterministic sequence
  // GT: [[1.0, 3.0], [2.0, 5.0]]
  // PRED: [[2.0, 1.0], [4.0, 2.0]]
  // Errors: [-1, 2, -2, 3] -> Abs: [1, 2, 2, 3] -> MAE = (1+2+2+3)/4 = 2.0
  // SqErrors: [1, 4, 4, 9] -> MSE = (1+4+4+9)/4 = 4.5
  std::vector<std::vector<double>> gt1 = { { 1.0, 3.0 }, { 2.0, 5.0 } };
  std::vector<std::vector<double>> pred1 = { { 2.0, 1.0 }, { 4.0, 2.0 } };

  EXPECT_DOUBLE_EQ(ErrorCalculation::calculate_mae_error(gt1, pred1), 2.0);
  EXPECT_DOUBLE_EQ(ErrorCalculation::calculate_mse_error(gt1, pred1), 4.5);

  // 2. RMSE on deterministic sequence
  // Seq 0 MSE = (1 + 4)/2 = 2.5 -> RMSE_0 = sqrt(2.5)
  // Seq 1 MSE = (4 + 9)/2 = 6.5 -> RMSE_1 = sqrt(6.5)
  // Avg RMSE = (sqrt(2.5) + sqrt(6.5)) / 2.0
  const double expected_rmse = (std::sqrt(2.5) + std::sqrt(6.5)) / 2.0;
  EXPECT_NEAR(ErrorCalculation::calculate_rmse_error(gt1, pred1), expected_rmse, 1e-12);

  // 3. Huber Loss with Delta = 1.0
  // GT: [[0.0], [0.0]]
  // PRED: [[0.5], [2.0]]
  // Error 0: 0.5 <= 1.0 -> 0.5 * 0.5^2 = 0.125
  // Error 1: 2.0 > 1.0  -> 1.0 * (2.0 - 0.5 * 1.0) = 1.5
  // Total = (0.125 + 1.5) / 2 = 0.8125
  std::vector<std::vector<double>> gt_huber = { { 0.0 }, { 0.0 } };
  std::vector<std::vector<double>> pred_huber = { { 0.5 }, { 2.0 } };
  EvaluationConfig huber_cfg(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12);
  EXPECT_DOUBLE_EQ(ErrorCalculation::calculate_huber_loss_error(gt_huber, pred_huber, huber_cfg), 0.8125);

  // 4. WAPE
  // GT: [[1.0, 3.0], [2.0, 4.0]]
  // PRED: [[1.5, 2.0], [3.0, 2.0]]
  // Abs error sum: |1-1.5| + |3-2| + |2-3| + |4-2| = 0.5 + 1.0 + 1.0 + 2.0 = 4.5
  // Abs actual sum: 1 + 3 + 2 + 4 = 10.0
  // WAPE = 4.5 / 10.0 = 0.45
  std::vector<std::vector<double>> gt_wape = { { 1.0, 3.0 }, { 2.0, 4.0 } };
  std::vector<std::vector<double>> pred_wape = { { 1.5, 2.0 }, { 3.0, 2.0 } };
  EXPECT_DOUBLE_EQ(ErrorCalculation::calculate_forecast_wape(gt_wape, pred_wape), 0.45);

  // 5. WAPE Zero Actuals Edge Case
  std::vector<std::vector<double>> gt_zero = { { 0.0, 0.0 } };
  std::vector<std::vector<double>> pred_zero = { { 0.0, 0.0 } };
  std::vector<std::vector<double>> pred_nonzero = { { 0.1, 0.2 } };
  EXPECT_DOUBLE_EQ(ErrorCalculation::calculate_forecast_wape(gt_zero, pred_zero), 0.0);
  EXPECT_DOUBLE_EQ(ErrorCalculation::calculate_forecast_wape(gt_zero, pred_nonzero), 1.0);

  // 6. BCE Loss
  // GT: [[1.0], [0.0]]
  // PRED: [[0.8], [0.2]]
  // Elem 0: -(1 * ln(0.8) + 0) = -ln(0.8)
  // Elem 1: -(0 + 1 * ln(0.8)) = -ln(0.8)
  // Avg = -ln(0.8)
  std::vector<std::vector<double>> gt_bce = { { 1.0 }, { 0.0 } };
  std::vector<std::vector<double>> pred_bce = { { 0.8 }, { 0.2 } };
  EvaluationConfig bce_cfg(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12);
  const double expected_bce = -std::log(0.8);
  EXPECT_NEAR(ErrorCalculation::calculate_bce_loss(gt_bce, pred_bce, bce_cfg), expected_bce, 1e-12);

  // 7. Log-Cosh
  // GT: [[0.0]]
  // PRED: [[1.0]]
  // x = 1.0
  // log_cosh = 1.0 + ln(1 + e^-2) - ln(2)
  std::vector<std::vector<double>> gt_logcosh = { { 0.0 } };
  std::vector<std::vector<double>> pred_logcosh = { { 1.0 } };
  const double expected_logcosh = 1.0 + std::log1p(std::exp(-2.0)) - std::log(2.0);
  EXPECT_NEAR(ErrorCalculation::calculate_log_cosh(gt_logcosh, pred_logcosh), expected_logcosh, 1e-12);
}

TEST_F(ErrorCalculationTest, MSESkipsNonFiniteValuesButKeepsFiniteOnesInTheAverage) {
  // A NaN/Inf prediction (e.g. from a diverged/unstable training run) must not corrupt the
  // whole metric: it should be excluded from both the sum and the count, rather than
  // poisoning the average with NaN or being silently counted as if it were a valid 0-error term.
  std::vector<std::vector<double>> gt = { { 1.0, 2.0, 3.0 } };
  std::vector<std::vector<double>> pred_with_nan = { { 2.0, std::numeric_limits<double>::quiet_NaN(), 3.0 } };
  std::vector<std::vector<double>> pred_with_inf = { { 2.0, std::numeric_limits<double>::infinity(), 3.0 } };

  // Index 1 (NaN) is skipped; the two remaining finite pairs are (1.0, 2.0) -> squared error
  // 1.0, and (3.0, 3.0) -> squared error 0.0, averaged over the 2 valid values: (1.0+0.0)/2.
  const double expected_finite_only = 0.5;
  EXPECT_DOUBLE_EQ(ErrorCalculation::calculate_mse_error(gt, pred_with_nan), expected_finite_only);
  EXPECT_DOUBLE_EQ(ErrorCalculation::calculate_mse_error(gt, pred_with_inf), expected_finite_only);
}

TEST_F(ErrorCalculationTest, MSEReturnsNaNWhenNoValidValuesExist) {
  std::vector<std::vector<double>> gt = { { 1.0, 2.0 } };
  std::vector<std::vector<double>> pred_all_nan = { { std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN() } };
  EXPECT_TRUE(std::isnan(ErrorCalculation::calculate_mse_error(gt, pred_all_nan)));

  // Mismatched row sizes are skipped with a warning rather than crashing; if every row
  // mismatches, no valid values exist at all.
  std::vector<std::vector<double>> gt_mismatch = { { 1.0, 2.0 } };
  std::vector<std::vector<double>> pred_mismatch = { { 1.0 } };
  EXPECT_TRUE(std::isnan(ErrorCalculation::calculate_mse_error(gt_mismatch, pred_mismatch)));
}

TEST_F(ErrorCalculationTest, MismatchedVectorSizePanicsForStrictMetrics) {
  std::vector<std::vector<double>> gt_mismatch = { { 1.0, 2.0 } };
  std::vector<std::vector<double>> pred_mismatch = { { 1.0 } };

  EXPECT_THROW((void)ErrorCalculation::calculate_mae_error(gt_mismatch, pred_mismatch), std::runtime_error);
  EXPECT_THROW((void)ErrorCalculation::calculate_huber_loss_error(gt_mismatch, pred_mismatch, config), std::runtime_error);
  EXPECT_THROW((void)ErrorCalculation::calculate_huber_direction_loss(gt_mismatch, pred_mismatch, config), std::runtime_error);
  EXPECT_THROW((void)ErrorCalculation::calculate_log_cosh(gt_mismatch, pred_mismatch), std::runtime_error);
  EXPECT_THROW((void)ErrorCalculation::calculate_directional_accuracy(gt_mismatch, pred_mismatch, config, activation::method::linear), std::runtime_error);
  EXPECT_THROW((void)ErrorCalculation::calculate_directional_confidence_score(gt_mismatch, pred_mismatch, config, activation::method::linear), std::runtime_error);

  std::vector<std::vector<double>> gt_sm_mismatch = { { 1.0, 0.0, 0.0 } };
  std::vector<std::vector<double>> pred_sm_mismatch = { { 0.5, 0.5 } };
  EXPECT_THROW((void)ErrorCalculation::calculate_softmax_directional_accuracy(gt_sm_mismatch, pred_sm_mismatch), std::runtime_error);
}

TEST_F(ErrorCalculationTest, MismatchedVectorSizeSkippedSilentlyForSequenceMetrics) {
  // rmse/nrmse/mape/smape/wape/bce/cross_entropy treat a mismatched row as "no data for
  // this sequence" and skip it, rather than panicking. Verify this documented, more forgiving
  // behaviour still holds (as opposed to accidentally throwing or corrupting the average).
  std::vector<std::vector<double>> gt = { { 1.0, 2.0 }, { 3.0 } };
  std::vector<std::vector<double>> pred = { { 1.5, 2.5 }, { 3.0, 4.0 } }; // row 1 mismatches

  EXPECT_NO_THROW((void)ErrorCalculation::calculate_rmse_error(gt, pred));
  EXPECT_NO_THROW((void)ErrorCalculation::calculate_nrmse_error(gt, pred));
  EXPECT_NO_THROW((void)ErrorCalculation::calculate_forecast_mape(gt, pred, config));
  EXPECT_NO_THROW((void)ErrorCalculation::calculate_forecast_smape(gt, pred, config));
  EXPECT_NO_THROW((void)ErrorCalculation::calculate_forecast_wape(gt, pred));
  EXPECT_NO_THROW((void)ErrorCalculation::calculate_bce_loss(gt, pred, config));
  EXPECT_NO_THROW((void)ErrorCalculation::calculate_cross_entropy(gt, pred, config));

  // Only row 0 contributes: RMSE = sqrt(((1-1.5)^2 + (2-2.5)^2) / 2) = sqrt(0.25) = 0.5
  EXPECT_NEAR(ErrorCalculation::calculate_rmse_error(gt, pred), 0.5, tolerance);
}

TEST_F(ErrorCalculationTest, UnknownStringToTypeThrows) {
  EXPECT_THROW((void)ErrorCalculation::string_to_type("not-a-real-error-type"), std::runtime_error);
}

TEST_F(ErrorCalculationTest, CalculateErrorEnumDispatch)
{
  std::vector<std::vector<double>> gt = { { 1.0, 0.0 }, { 0.0, 1.0 } };
  std::vector<std::vector<double>> pred = { { 0.8, 0.2 }, { 0.3, 0.7 } };

  EXPECT_DOUBLE_EQ(ErrorCalculation::calculate_error(ErrorCalculation::type::none, gt, pred, config, activation::method::sigmoid), 0.0);
  EXPECT_GT(ErrorCalculation::calculate_error(ErrorCalculation::type::mae, gt, pred, config, activation::method::sigmoid), 0.0);
  EXPECT_GT(ErrorCalculation::calculate_error(ErrorCalculation::type::mse, gt, pred, config, activation::method::sigmoid), 0.0);
  EXPECT_GT(ErrorCalculation::calculate_error(ErrorCalculation::type::rmse, gt, pred, config, activation::method::sigmoid), 0.0);
  EXPECT_GT(ErrorCalculation::calculate_error(ErrorCalculation::type::bce_loss, gt, pred, config, activation::method::sigmoid), 0.0);
  EXPECT_GT(ErrorCalculation::calculate_error(ErrorCalculation::type::cross_entropy, gt, pred, config, activation::method::softmax), 0.0);
}

