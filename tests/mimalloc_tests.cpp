#include <gtest/gtest.h>
#include "common/aligned_allocator.h"
#include <vector>
#include <thread>
#include <atomic>
#include <cstdint>

#if defined(MYODDWEB_USE_MIMALLOC)
#include "../include/neuralnetwork/libraries/mimalloc/include/mimalloc.h"
#endif

using namespace myoddweb::nn;

namespace
{
void thread_allocation_worker(size_t iterations, size_t alloc_size, std::atomic<size_t>& success_count)
{
  for (size_t i = 0; i < iterations; ++i)
  {
    AlignedAllocator<double, 32> allocator;
    double* buffer = allocator.allocate(alloc_size);
    if (buffer != nullptr)
    {
      for (size_t j = 0; j < alloc_size; ++j)
      {
        buffer[j] = static_cast<double>(i + j);
      }
      for (size_t j = 0; j < alloc_size; ++j)
      {
        if (buffer[j] != static_cast<double>(i + j))
        {
          allocator.deallocate(buffer, alloc_size);
          return;
        }
      }
      allocator.deallocate(buffer, alloc_size);
      success_count.fetch_add(1, std::memory_order_relaxed);
    }
  }
}
} // namespace

TEST(MimallocTest, BuildConfigurationActive)
{
#if defined(MYODDWEB_USE_MIMALLOC)
  const int version = mi_version();
  EXPECT_GT(version, 0);
#else
  SUCCEED() << "mimalloc is disabled in this build configuration";
#endif
}

TEST(MimallocTest, AlignedAllocationVariousAlignments)
{
  // Test 16, 32, 64, 128 byte alignments
  {
    AlignedAllocator<double, 16> alloc16;
    double* p16 = alloc16.allocate(64);
    ASSERT_NE(p16, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p16) % 16, 0u);
    alloc16.deallocate(p16, 64);
  }

  {
    AlignedAllocator<double, 32> alloc32;
    double* p32 = alloc32.allocate(64);
    ASSERT_NE(p32, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p32) % 32, 0u);
    alloc32.deallocate(p32, 64);
  }

  {
    AlignedAllocator<double, 64> alloc64;
    double* p64 = alloc64.allocate(64);
    ASSERT_NE(p64, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p64) % 64, 0u);
    alloc64.deallocate(p64, 64);
  }

  {
    AlignedAllocator<double, 128> alloc128;
    double* p128 = alloc128.allocate(64);
    ASSERT_NE(p128, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p128) % 128, 0u);
    alloc128.deallocate(p128, 64);
  }
}

TEST(MimallocTest, AlignedVectorLifecycle)
{
  AlignedVector<double, 32> vec;
  const size_t test_size = 256;
  vec.resize(test_size);

  EXPECT_EQ(reinterpret_cast<uintptr_t>(vec.data()) % 32, 0u);

  for (size_t i = 0; i < test_size; ++i)
  {
    vec[i] = static_cast<double>(i * 2);
  }

  for (size_t i = 0; i < test_size; ++i)
  {
    EXPECT_DOUBLE_EQ(vec[i], static_cast<double>(i * 2));
  }
}

TEST(MimallocTest, MultiThreadedConcurrentAllocations)
{
  const size_t num_threads = 8;
  const size_t iterations_per_thread = 500;
  const size_t alloc_size = 128;
  std::atomic<size_t> success_count{ 0 };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (size_t t = 0; t < num_threads; ++t)
  {
    threads.emplace_back(thread_allocation_worker, iterations_per_thread, alloc_size, std::ref(success_count));
  }

  for (size_t t = 0; t < num_threads; ++t)
  {
    if (threads[t].joinable())
    {
      threads[t].join();
    }
  }

  EXPECT_EQ(success_count.load(), num_threads * iterations_per_thread);
}
