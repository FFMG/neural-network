#include <gtest/gtest.h>
#include "common/taskqueue.h"
#include <atomic>
#include <chrono>
#include <vector>
#include <thread>


using namespace myoddweb::nn;
class TaskQueueComprehensiveTest : public ::testing::Test 
{
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TaskQueueComprehensiveTest, TaskQueueBasicExecution) 
{
  TaskQueue<int> queue;
  for (int i = 0; i < 10; ++i) 
  {
    queue.enqueue([i]() { return i * 2; });
  }
  
  auto results = queue.get();
  ASSERT_EQ(results.size(), 10);
  
  // Results might be out of order depending on how get() collects local_results,
  // but in TaskQueue they are collected in the order they finish.
  // Actually, TaskQueue uses local_results.push_back then inserts at the end.
  // We just check if all values are present.
  std::sort(results.begin(), results.end());
  for (int i = 0; i < 10; ++i) 
  {
    EXPECT_EQ(results[i], i * 2);
  }
}

TEST_F(TaskQueueComprehensiveTest, TaskQueueVoidExecution) 
{
  TaskQueue<void> queue;
  std::atomic<int> counter{0};
  for (int i = 0; i < 10; ++i) 
  {
    queue.enqueue([&counter]() { counter.fetch_add(1); });
  }
  queue.get();
  EXPECT_EQ(counter.load(), 10);
}

TEST_F(TaskQueueComprehensiveTest, SingleTaskQueueBasicExecution) 
{
  SingleTaskQueue<int> queue;
  EXPECT_TRUE(queue.call([]() { return 42; }));
  EXPECT_EQ(queue.get(), 42);
}

TEST_F(TaskQueueComprehensiveTest, SingleTaskQueueBusy) 
{
  SingleTaskQueue<void> queue;
  std::atomic<bool> start{false};
  std::atomic<bool> done{false};
  
  queue.call([&start, &done]() 
  {
    start.store(true);
    while (!done.load()) 
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });

  // Wait for it to start
  while(!start.load()) std::this_thread::yield();

  EXPECT_TRUE(queue.busy());
  EXPECT_FALSE(queue.call([]() {})); // Should fail as it is busy
  
  done.store(true);
  queue.get();
  EXPECT_FALSE(queue.busy());
}

TEST_F(TaskQueueComprehensiveTest, TaskQueueStressTest) 
{
  // This test aims to trigger the race condition where the worker thread might exit.
  for (int i = 0; i < 100; ++i) 
  {
    TaskQueue<void> queue;
    std::atomic<bool> executed{false};
    queue.enqueue([&executed]() { executed.store(true); });
    queue.get();
    EXPECT_TRUE(executed.load()) << "Failed at iteration " << i;
  }
}

TEST_F(TaskQueueComprehensiveTest, SingleTaskQueueStressTest) 
{
  for (int i = 0; i < 100; ++i) 
  {
    SingleTaskQueue<void> queue;
    std::atomic<bool> executed{false};
    queue.call([&executed]() { executed.store(true); });
    queue.get();
    EXPECT_TRUE(executed.load()) << "Failed at iteration " << i;
  }
}

TEST_F(TaskQueueComprehensiveTest, TaskQueueExceptionHandling) 
{
  TaskQueue<int> queue;
  queue.enqueue([]() -> int { throw std::runtime_error("Test Exception"); });
  EXPECT_THROW(queue.get(), std::runtime_error);
}

TEST_F(TaskQueueComprehensiveTest, SingleTaskQueueExceptionHandling) 
{
  SingleTaskQueue<int> queue;
  queue.call([]() -> int { throw std::runtime_error("Test Exception"); });
  EXPECT_THROW(queue.get(), std::runtime_error);
}
