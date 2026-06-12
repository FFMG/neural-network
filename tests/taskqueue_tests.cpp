#include <gtest/gtest.h>
#include "../src/neuralnetwork/taskqueue.h"
#include <atomic>
#include <thread>
#include <vector>
#include <numeric>
#include <chrono>

// --- TaskQueue<int> Tests ---


using namespace myoddweb::nn;
TEST(TaskQueueTest, BasicIntOutput) {
    TaskQueue<int> queue;
    for (int i = 0; i < 10; ++i) {
        queue.enqueue([i]() { return i * 2; });
    }
    auto results = queue.get();
    ASSERT_EQ(results.size(), 10);
    int sum = std::accumulate(results.begin(), results.end(), 0);
    EXPECT_EQ(sum, 90); // 0+2+4+6+8+10+12+14+16+18 = 90
}

TEST(TaskQueueTest, ConcurrencyBombardment) {
    TaskQueue<int> queue;
    std::atomic<int> producers_finished{0};
    const int num_producers = 10;
    const int tasks_per_producer = 1000;
    std::vector<std::thread> producers;

    for (int p = 0; p < num_producers; ++p) {
        producers.emplace_back([&queue, &producers_finished]() {
            for (int i = 0; i < tasks_per_producer; ++i) {
                queue.enqueue([]() { return 1; });
            }
            producers_finished++;
        });
    }

    for (auto& t : producers) t.join();
    
    auto results = queue.get();
    EXPECT_EQ(results.size(), num_producers * tasks_per_producer);
    int total = std::accumulate(results.begin(), results.end(), 0);
    EXPECT_EQ(total, num_producers * tasks_per_producer);
}

TEST(TaskQueueTest, ExceptionPropagation) {
    TaskQueue<int> queue;
    queue.enqueue([]() -> int {
        throw std::runtime_error("Task failure");
    });
    EXPECT_THROW(queue.get(), std::runtime_error);
}

// --- TaskQueue<void> Tests ---

TEST(TaskQueueVoidTest, BasicSideEffect) {
    TaskQueue<void> queue;
    std::atomic<int> counter{0};
    for (int i = 0; i < 50; ++i) {
        queue.enqueue([&counter]() { counter++; });
    }
    queue.get();
    EXPECT_EQ(counter.load(), 50);
}

TEST(TaskQueueVoidTest, StopAndJoinRace) {
    // Stress test the stop() logic we fixed
    for (int i = 0; i < 100; ++i) {
        TaskQueue<void> queue;
        std::atomic<int> processed{0};
        queue.enqueue([&processed]() { 
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            processed++; 
        });
        // Multiple threads calling stop() simultaneously
        std::thread t1([&queue]() { queue.stop(); });
        std::thread t2([&queue]() { queue.stop(); });
        t1.join();
        t2.join();
    }
}

// --- SingleTaskQueue Tests ---

TEST(SingleTaskQueueTest, BusyStatus) {
    SingleTaskQueue<int> queue;
    std::atomic<bool> can_finish{false};
    
    // Start a long running task
    bool called = queue.call([&can_finish]() {
        while(!can_finish.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        return 42;
    });
    EXPECT_TRUE(called);
    EXPECT_TRUE(queue.busy());

    // Try to call another one - should fail
    bool called_again = queue.call([]() { return 0; });
    EXPECT_FALSE(called_again);

    can_finish.store(true);
    int result = queue.get();
    EXPECT_EQ(result, 42);
    EXPECT_FALSE(queue.busy());
}

TEST(SingleTaskQueueVoidTest, ExceptionCatching) {
    SingleTaskQueue<void> queue;
    queue.call([]() {
        throw std::logic_error("Single failure");
    });
    EXPECT_THROW(queue.get(), std::logic_error);
}

// --- TaskQueuePool Tests ---

TEST(TaskQueuePoolTest, DistributionStress) {
    TaskQueuePool<int> pool(4);
    const int total_tasks = 4000;
    for (int i = 0; i < total_tasks; ++i) {
        pool.enqueue([]() { return 1; });
    }
    auto results = pool.get();
    EXPECT_EQ(results.size(), total_tasks);
    int total = std::accumulate(results.begin(), results.end(), 0);
    EXPECT_EQ(total, total_tasks);
}

TEST(TaskQueuePoolVoidTest, RapidLifecycle) {
    for (int i = 0; i < 20; ++i) {
        TaskQueuePool<void> pool(8);
        std::atomic<int> counter{0};
        for (int j = 0; j < 100; ++j) {
            pool.enqueue([&counter]() { counter++; });
        }
        pool.get();
        EXPECT_EQ(counter.load(), 100);
        pool.stop();
    }
}

TEST(TaskQueuePoolTest, MultipleGets) {
    TaskQueuePool<int> pool(2);
    pool.enqueue([]() { return 10; });
    auto r1 = pool.get();
    ASSERT_EQ(r1.size(), 1);
    EXPECT_EQ(r1[0], 10);

    pool.enqueue([]() { return 20; });
    pool.enqueue([]() { return 30; });
    auto r2 = pool.get();
    ASSERT_EQ(r2.size(), 2);
    // Note: order is not guaranteed across different internal queues, 
    // but results should be there.
    int sum = r2[0] + r2[1];
    EXPECT_EQ(sum, 50);
}
