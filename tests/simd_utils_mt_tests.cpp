#include <gtest/gtest.h>
#include "common/simd_utils.h"
#include <vector>
#include <thread>
#include <atomic>


using namespace myoddweb::nn;
namespace {
  constexpr size_t NUM_THREADS = 8;
  constexpr size_t NUM_ITERATIONS = 1000;
  constexpr size_t VECTOR_SIZE = 1024;
}

TEST(SimdUtilsMtTest, ConcurrentMulAdd) {
  std::vector<std::thread> threads;
  std::atomic<bool> start{ false };

  for (size_t i = 0; i < NUM_THREADS; ++i) {
    threads.emplace_back([&start]() {
      while (!start) { std::this_thread::yield(); }
      
      std::vector<double> w(VECTOR_SIZE, 1.0);
      std::vector<double> y(VECTOR_SIZE, 0.0);
      for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        simd::mul_add(0.5, w.data(), y.data(), VECTOR_SIZE);
      }
      
      for (double val : y) {
        EXPECT_NEAR(val, 0.5 * NUM_ITERATIONS, 1e-9);
      }
    });
  }

  start = true;
  for (auto& t : threads) {
    t.join();
  }
}

TEST(SimdUtilsMtTest, ConcurrentDotProduct) {
  std::vector<std::thread> threads;
  std::atomic<bool> start{ false };

  for (size_t i = 0; i < NUM_THREADS; ++i) {
    threads.emplace_back([&start]() {
      while (!start) { std::this_thread::yield(); }

      std::vector<double> a(VECTOR_SIZE, 0.1);
      std::vector<double> b(VECTOR_SIZE, 0.2);
      for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        double dot = simd::dot_product(a.data(), b.data(), VECTOR_SIZE);
        EXPECT_NEAR(dot, 0.1 * 0.2 * VECTOR_SIZE, 1e-9);
      }
    });
  }

  start = true;
  for (auto& t : threads) {
    t.join();
  }
}

TEST(SimdUtilsMtTest, ConcurrentScalarMulAdd) {
  std::vector<std::thread> threads;
  std::atomic<bool> start{ false };

  for (size_t i = 0; i < NUM_THREADS; ++i) {
    threads.emplace_back([&start]() {
      while (!start) { std::this_thread::yield(); }

      std::vector<double> w(VECTOR_SIZE, 1.0);
      std::vector<double> y(VECTOR_SIZE, 0.0);
      for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        simd::scalar_mul_add(0.5, w.data(), y.data(), VECTOR_SIZE);
      }

      for (double val : y) {
        EXPECT_NEAR(val, 0.5 * NUM_ITERATIONS, 1e-9);
      }
    });
  }

  start = true;
  for (auto& t : threads) {
    t.join();
  }
}
