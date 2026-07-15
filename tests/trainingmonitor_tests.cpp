#include <gtest/gtest.h>
#include "helpers/trainingmonitor.h"

using namespace myoddweb::nn;

TEST(TrainingMonitorTest, DefaultConstructor)
{
  TrainingMonitor monitor;
  // By default, evaluate should print error and return OnTrack because metrics are missing
  EXPECT_EQ(monitor.evaluate(), TrainingMonitor::TrainingStatus::OnTrack);
}

TEST(TrainingMonitorTest, CopyAndMoveOperations)
{
  TrainingMonitor monitor(5, 0.6, 0.4, 0.52, 1e-3);
  monitor.add_metric(ErrorCalculation::type::rmse, 0.5);
  monitor.add_metric(ErrorCalculation::type::directional_accuracy, 0.6);

  // Copy constructor
  TrainingMonitor copy_monitor(monitor);
  EXPECT_EQ(copy_monitor.evaluate(), TrainingMonitor::TrainingStatus::OnTrack);

  // Move constructor
  TrainingMonitor move_monitor(std::move(monitor));
  EXPECT_EQ(move_monitor.evaluate(), TrainingMonitor::TrainingStatus::OnTrack);

  // Copy assignment
  TrainingMonitor copy_assign;
  copy_assign = copy_monitor;
  EXPECT_EQ(copy_assign.evaluate(), TrainingMonitor::TrainingStatus::OnTrack);

  // Move assignment
  TrainingMonitor move_assign;
  move_assign = std::move(copy_monitor);
  EXPECT_EQ(move_assign.evaluate(), TrainingMonitor::TrainingStatus::OnTrack);
}

TEST(TrainingMonitorTest, AddMetricHistoryBounding)
{
  // window size 3, max size = 3 * 5 = 15
  TrainingMonitor monitor(3);
  
  // Add 20 metrics
  for (int i = 0; i < 20; ++i)
  {
    monitor.add_metric(ErrorCalculation::type::rmse, static_cast<double>(i));
  }

  // We cannot query the history directly since the member is private,
  // but we can ensure that adding metrics doesn't crash and bounds correctly.
  // Let's add enough metrics for evaluate
  for (int i = 0; i < 20; ++i)
  {
    monitor.add_metric(ErrorCalculation::type::directional_accuracy, 0.8);
  }

  EXPECT_EQ(monitor.evaluate(), TrainingMonitor::TrainingStatus::Diverging);
}

TEST(TrainingMonitorTest, StatusOnTrack)
{
  TrainingMonitor monitor(3, 0.5, 0.5, 0.5, 1e-3);

  // RMSE decreasing (improving), DA increasing (improving)
  monitor.add_metric(ErrorCalculation::type::rmse, 0.5);
  monitor.add_metric(ErrorCalculation::type::rmse, 0.3);
  monitor.add_metric(ErrorCalculation::type::rmse, 0.1); // slope: -0.2

  monitor.add_metric(ErrorCalculation::type::directional_accuracy, 0.6);
  monitor.add_metric(ErrorCalculation::type::directional_accuracy, 0.7);
  monitor.add_metric(ErrorCalculation::type::directional_accuracy, 0.8); // slope: 0.1

  // Score: -(-0.2 * 0.5) + (0.1 * 0.5) = 0.1 + 0.05 = 0.15 (improving > 0.01, DA > 0.5)
  EXPECT_EQ(monitor.evaluate(), TrainingMonitor::TrainingStatus::OnTrack);
}

TEST(TrainingMonitorTest, StatusDiverging)
{
  TrainingMonitor monitor(3, 0.5, 0.5, 0.5, 1e-3);

  // RMSE increasing (getting worse), DA decreasing (getting worse)
  monitor.add_metric(ErrorCalculation::type::rmse, 0.1);
  monitor.add_metric(ErrorCalculation::type::rmse, 0.3);
  monitor.add_metric(ErrorCalculation::type::rmse, 0.5); // slope: 0.2

  monitor.add_metric(ErrorCalculation::type::directional_accuracy, 0.8);
  monitor.add_metric(ErrorCalculation::type::directional_accuracy, 0.7);
  monitor.add_metric(ErrorCalculation::type::directional_accuracy, 0.6); // slope: -0.1

  // Score: -(0.2 * 0.5) + (-0.1 * 0.5) = -0.1 - 0.05 = -0.15 (< 0.0)
  EXPECT_EQ(monitor.evaluate(), TrainingMonitor::TrainingStatus::Diverging);
}

TEST(TrainingMonitorTest, StatusStuckDueToLowAccuracy)
{
  TrainingMonitor monitor(3, 0.5, 0.5, 0.6, 1e-3); // threshold = 0.6

  // RMSE decreasing (improving), but DA is flat at 0.5 (below threshold 0.6)
  monitor.add_metric(ErrorCalculation::type::rmse, 0.5);
  monitor.add_metric(ErrorCalculation::type::rmse, 0.3);
  monitor.add_metric(ErrorCalculation::type::rmse, 0.1); // slope: -0.2

  monitor.add_metric(ErrorCalculation::type::directional_accuracy, 0.5);
  monitor.add_metric(ErrorCalculation::type::directional_accuracy, 0.5);
  monitor.add_metric(ErrorCalculation::type::directional_accuracy, 0.5); // slope: 0.0

  // Score: -(-0.2 * 0.5) + (0.0 * 0.5) = 0.1
  // However, DA is 0.5 which is < 0.6 threshold, so it should be Stuck
  EXPECT_EQ(monitor.evaluate(), TrainingMonitor::TrainingStatus::Stuck);
}

TEST(TrainingMonitorTest, StatusStuckDueToLowScore)
{
  TrainingMonitor monitor(3, 0.5, 0.5, 0.5, 1e-3);

  // RMSE flat, DA flat (no improvement)
  monitor.add_metric(ErrorCalculation::type::rmse, 0.2);
  monitor.add_metric(ErrorCalculation::type::rmse, 0.2);
  monitor.add_metric(ErrorCalculation::type::rmse, 0.2); // slope: 0.0

  monitor.add_metric(ErrorCalculation::type::directional_accuracy, 0.8);
  monitor.add_metric(ErrorCalculation::type::directional_accuracy, 0.8);
  monitor.add_metric(ErrorCalculation::type::directional_accuracy, 0.8); // slope: 0.0

  // Score: 0.0, which is < 0.01, so it should be Stuck
  EXPECT_EQ(monitor.evaluate(), TrainingMonitor::TrainingStatus::Stuck);
}

TEST(TrainingMonitorTest, StatusToString)
{
  EXPECT_STREQ(TrainingMonitor::monitor_status_to_string(TrainingMonitor::TrainingStatus::OnTrack), "on-track");
  EXPECT_STREQ(TrainingMonitor::monitor_status_to_string(TrainingMonitor::TrainingStatus::Stuck), "stuck");
  EXPECT_STREQ(TrainingMonitor::monitor_status_to_string(TrainingMonitor::TrainingStatus::Diverging), "diverging");
  EXPECT_STREQ(TrainingMonitor::monitor_status_to_string(static_cast<TrainingMonitor::TrainingStatus>(99)), "unknown");
}
