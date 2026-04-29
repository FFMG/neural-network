#include <gtest/gtest.h>
#include "errorcalculation.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace math_expect {
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

  double log_cosh(const std::vector<std::vector<double>>& gt, const std::vector<std::vector<double>>& pred) {
    double total = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
      for (size_t j = 0; j < gt[i].size(); ++j) {
        double x = std::abs(gt[i][j] - pred[i][j]);
        // ln(cosh(x)) = x + ln(1 + e^(-2x)) - ln(2)
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
}

class ErrorCalculationTest : public ::testing::Test {
protected:
  std::vector<std::vector<double>> ground_truth = {
    {1.0, 0.5, -0.2},
    {0.0, 1.0, 0.8}
  };
  std::vector<std::vector<double>> predictions = {
    {0.9, 0.6, -0.1},
    {0.2, 0.8, 0.7}
  };
  EvaluationConfig config{0.05, 0.5, 1.0, 0.1, true, 1.0, 1e-7};
  double tolerance = 1e-9;
};

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

TEST_F(ErrorCalculationTest, Huber) {
  double expected = math_expect::huber(ground_truth, predictions, config.huber_delta());
  double actual = ErrorCalculation::calculate_huber_loss_error(ground_truth, predictions, config);
  EXPECT_NEAR(expected, actual, tolerance);
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

TEST_F(ErrorCalculationTest, WAPE) {
  double expected = math_expect::wape(ground_truth, predictions);
  double actual = ErrorCalculation::calculate_forecast_wape(ground_truth, predictions);
  EXPECT_NEAR(expected, actual, tolerance);
}

TEST_F(ErrorCalculationTest, sMAPE) {
  double expected = math_expect::smape(ground_truth, predictions, config.epsilon());
  double actual = ErrorCalculation::calculate_forecast_smape(ground_truth, predictions, config);
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

TEST_F(ErrorCalculationTest, DirectionalAccuracy) {
  // Test with Sigmoid baseline (0.5)
  std::vector<std::vector<double>> gt = {{0.6, 0.4, 0.5}}; // UP, DOWN, NEUTRAL
  std::vector<std::vector<double>> pr = {{0.7, 0.3, 0.8}}; // UP(Match), DOWN(Match), UP(Ignore)
  
  // Neutral tolerance is 0.05. 0.5 is neutral. 0.6 is UP, 0.4 is DOWN.
  // Sequence 1: 
  // 0.6 vs 0.7 -> UP vs UP -> Match
  // 0.4 vs 0.3 -> DOWN vs DOWN -> Match
  // 0.5 vs 0.8 -> NEUTRAL vs UP -> Ignore
  // Total = 2, Correct = 2 -> 1.0
  
  double actual = ErrorCalculation::calculate_directional_accuracy(gt, pr, config, activation::method::sigmoid);
  EXPECT_DOUBLE_EQ(actual, 1.0);
  
  // Test with Tanh baseline (0.0)
  std::vector<std::vector<double>> gt2 = {{0.1, -0.1, 0.0}}; // UP, DOWN, NEUTRAL
  std::vector<std::vector<double>> pr2 = {{0.2, 0.1, -0.5}}; // UP(Match), UP(Fail), DOWN(Ignore)
  // Total = 2, Correct = 1 -> 0.5
  double actual2 = ErrorCalculation::calculate_directional_accuracy(gt2, pr2, config, activation::method::tanh);
  EXPECT_DOUBLE_EQ(actual2, 0.5);
}

TEST_F(ErrorCalculationTest, SoftmaxAccuracy) {
  // Softmax uses ArgMax
  std::vector<std::vector<double>> gt = {{0.0, 1.0, 0.0}}; // Class 1 (UP if mid=1)
  std::vector<std::vector<double>> pr = {{0.1, 0.8, 0.1}}; // Class 1 (Match)
  
  // mid = (3-1)/2 = 1.0
  // gt_idx = 1 (Neutral! Index 1 == mid) -> Should be ignored if we use strict neutral check?
  // Let's check code: const bool is_gt_neutral = std::abs(static_cast<double>(gt_idx) - mid) < 0.1;
  // Yes, index 1 is neutral.
  
  std::vector<std::vector<double>> gt2 = {{1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}}; // Class 0 (DOWN), Class 2 (UP)
  std::vector<std::vector<double>> pr2 = {{0.9, 0.1, 0.0}, {0.4, 0.1, 0.5}}; // Class 0 (Match), Class 2 (Match)
  // Total = 2, Correct = 2 -> 1.0
  double actual = ErrorCalculation::calculate_softmax_directional_accuracy(gt2, pr2, config);
  EXPECT_DOUBLE_EQ(actual, 1.0);
}

TEST_F(ErrorCalculationTest, HuberDirectionLoss) {
  // Huber Direction Loss = Huber + lambda * ln(1 + exp(-100 * target * output))
  // Sequence 1, Neuron 1: target=1.0, output=0.9
  // error = 0.1, abs_error = 0.1 <= delta(1.0)
  // huber_part = 0.5 * 0.1 * 0.1 = 0.005
  // x = -100 * 1.0 * 0.9 = -90.0
  // direction_part = ln(1 + exp(-90)) approx 0
  // loss = 0.005 + 0.1 * 0 = 0.005
  
  std::vector<std::vector<double>> gt = {{1.0}};
  std::vector<std::vector<double>> pr = {{0.9}};
  double actual = ErrorCalculation::calculate_huber_direction_loss(gt, pr, config);
  EXPECT_NEAR(actual, 0.005, 1e-12);

  // Sign mismatch: target=1.0, output=-0.1
  // error = -1.1, abs_error = 1.1 > delta(1.0)
  // huber_part = 1.0 * (1.1 - 0.5 * 1.0) = 0.6
  // x = -100 * 1.0 * -0.1 = 10.0
  // direction_part = ln(1 + exp(10)) = 10 + ln(1 + exp(-10)) approx 10.000045
  // loss = 0.6 + 0.1 * 10.000045398 = 1.6000045...
  double actual_mismatch = ErrorCalculation::calculate_huber_direction_loss(gt, std::vector<std::vector<double>>{{-0.1}}, config);
  EXPECT_NEAR(actual_mismatch, 0.6 + 0.1 * (10.0 + std::log1p(std::exp(-10.0))), 1e-12);
}

TEST_F(ErrorCalculationTest, NRMSE) {
  // NRMSE averages (RMSE / (max-min or mean_abs)) per sequence
  // Seq 1: gt={1.0, 0.5, -0.2}, pr={0.9, 0.6, -0.1}
  // mse = ((0.1)^2 + (-0.1)^2 + (-0.1)^2) / 3 = 0.03 / 3 = 0.01
  // rmse = 0.1
  // max=1.0, min=-0.2, denom=1.2
  // nrmse_1 = 0.1 / 1.2 = 0.0833333333
  
  // Seq 2: gt={0.0, 1.0, 0.8}, pr={0.2, 0.8, 0.7}
  // mse = ((-0.2)^2 + (0.2)^2 + (0.1)^2) / 3 = (0.04 + 0.04 + 0.01) / 3 = 0.03
  // rmse = sqrt(0.03) approx 0.173205
  // max=1.0, min=0.0, denom=1.0
  // nrmse_2 = 0.173205
  
  // Final = (0.08333333 + 0.173205) / 2 approx 0.128269
  double actual = ErrorCalculation::calculate_nrmse_error(ground_truth, predictions);
  double nrmse1 = 0.1 / 1.2;
  double nrmse2 = std::sqrt(0.03) / 1.0;
  EXPECT_NEAR(actual, (nrmse1 + nrmse2) / 2.0, 1e-7);
}

TEST_F(ErrorCalculationTest, Coverage) {
  // Prediction Coverage: % of samples where ANY neuron is > confidence_threshold(0.5)
  // Seq 1: {0.9, 0.6, -0.1} -> Confident (0.9 > 0.5)
  // Seq 2: {0.2, 0.3, 0.4} -> Not Confident (all <= 0.5)
  std::vector<std::vector<double>> pr = {{0.9, 0.6, -0.1}, {0.2, 0.3, 0.4}};
  double actual = ErrorCalculation::calculate_prediction_coverage(pr, config, activation::method::linear);
  EXPECT_DOUBLE_EQ(actual, 0.5);

  // Softmax Coverage: winning class > threshold
  // Seq 1: {0.1, 0.8, 0.1} -> Confident (0.8 > 0.5)
  // Seq 2: {0.4, 0.3, 0.3} -> Not Confident (0.4 <= 0.5)
  double actual_sm = ErrorCalculation::calculate_softmax_prediction_coverage(pr, config);
  EXPECT_DOUBLE_EQ(actual_sm, 0.5);
}

TEST_F(ErrorCalculationTest, DirectionalConfidenceScore) {
  // Score = Directional Accuracy filtered by confidence_threshold(0.5)
  // gt = {{1.0, -1.0, 1.0}} (UP, DOWN, UP)
  // pr = {{0.9, -0.1, -0.8}} (UP(Match, Confident), DOWN(Ignore, Not Confident), DOWN(Fail, Confident))
  // Total confident & not neutral = 2 (index 0 and 2)
  // Correct = 1 (index 0)
  // Result = 0.5
  std::vector<std::vector<double>> gt = {{1.0, -1.0, 1.0}};
  std::vector<std::vector<double>> pr = {{0.9, -0.1, -0.8}};
  double actual = ErrorCalculation::calculate_directional_confidence_score(gt, pr, config, activation::method::tanh);
  EXPECT_DOUBLE_EQ(actual, 0.5);
}

TEST_F(ErrorCalculationTest, SoftmaxDirectionalConfidence) {
  // mid = 1.0 (for 3 classes)
  // gt = {{1, 0, 0}} (Index 0: DOWN)
  // pr = {{0.6, 0.2, 0.2}} (Index 0: DOWN, Confidence 0.6 > 0.5) -> Match
  std::vector<std::vector<double>> gt = {{1.0, 0.0, 0.0}};
  std::vector<std::vector<double>> pr = {{0.6, 0.2, 0.2}};
  double actual = ErrorCalculation::calculate_softmax_directional_confidence_score(gt, pr, config);
  EXPECT_DOUBLE_EQ(actual, 1.0);

  // pr = {{0.4, 0.4, 0.2}} (Index 0: DOWN, Confidence 0.4 <= 0.5) -> Ignore
  std::vector<std::vector<double>> pr2 = {{0.4, 0.4, 0.2}};
  double actual2 = ErrorCalculation::calculate_softmax_directional_confidence_score(gt, pr2, config);
  EXPECT_DOUBLE_EQ(actual2, 0.0);
}


