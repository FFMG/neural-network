// neuralnetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "logger.h"
#include "taskqueue.h"
#include <chrono> // For std::chrono::milliseconds
#include <iostream>
#include <atomic>
#include <mutex>

#include "./examples/addingproblem.h"
#include "./examples/copymemory.h"
#include "./examples/residualxor.h"
#include "./examples/spiral.h"
#include "./examples/threebitparity.h"
#include "./examples/twomoon.h"
#include "./examples/xor.h"
#include "./libraries/instrumentor.h"

// Fibonacci function for simulating work
long long fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Test function for TaskQueuePool<void> deadlock detection
void test_task_queue_pool_deadlock(int num_threads, int num_tasks) {
    Logger::info(Logger::factory("Running TaskQueuePool deadlock test with ", num_threads, " threads and ", num_tasks, " tasks."));
    TaskQueuePool<void> pool(num_threads);
    std::atomic<int> completed_tasks_count(0);
    std::mutex cout_mutex; // For protecting Logger output

    for (int i = 0; i < num_tasks; ++i) {
        pool.enqueue([&, i]() { // Capture by reference for shared variables, by value for loop variable i
            long long result = fibonacci(10 + (i % 5)); // Vary the input to fibonacci slightly
            completed_tasks_count.fetch_add(1);
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                Logger::debug(Logger::factory("Task ", i, " completed. Result (Fibonacci): ", result));
            }
        });
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Wait for all tasks to complete
    pool.get();

    // Verify all tasks completed
    if (completed_tasks_count.load() == num_tasks) {
        Logger::info(Logger::factory("TaskQueuePool deadlock test PASSED. All ", completed_tasks_count.load(), " tasks completed."));
    } else {
        Logger::error(Logger::factory("TaskQueuePool deadlock test FAILED. Expected ", num_tasks, " tasks, but completed ", completed_tasks_count.load(), "."));
    }
}

int main()
{
  MYODDWEB_PROFILE_BEGIN_SESSION( "Monitor Global", "Profile-Global.json" );

  auto log_level = Logger::LogLevel::Trace;
  Logger::set_level(log_level);

  // Run the TaskQueuePool deadlock test
  // test_task_queue_pool_deadlock(11, 1000); // Test with 4 threads and 100 tasks
  // You can uncomment and test with different values, e.g., to stress test:
  // test_task_queue_pool_deadlock(std::thread::hardware_concurrency(), 1000);

  // Copy Memory
  // ExampleCopyMemory::MemoryCopy(log_level);

  // Spiral
  // ExampleSpiral::Spiral(log_level, true);

  // Two Moon
  // ExampleTwoMoon::TwoMoon(log_level, true);

  // Adding Problem
  ExampleAddingProblem::AddingProblem(log_level);

  // XOR
  // ExampleXor::Xor(log_level, true);

  // Residual XOR
  // ExampleResidualXor::Xor(log_level, true);

  // 3-bit Parity
  // ExampleThreebitParity::ThreebitParity(log_level);

  MYODDWEB_PROFILE_END_SESSION();

  return 0;
}
