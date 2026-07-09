#include <gtest/gtest.h>
#include "common/aligned_allocator.h"
#include <vector>
#include <cstdint>

// A dummy class to test construct/destroy

using namespace myoddweb::nn;
class LifecycleTracker {
public:
    static int constructions;
    static int destructions;
    LifecycleTracker() { constructions++; }
    LifecycleTracker(const LifecycleTracker&) { constructions++; }
    LifecycleTracker(LifecycleTracker&&) noexcept { constructions++; }
    ~LifecycleTracker() { destructions++; }
};
int LifecycleTracker::constructions = 0;
int LifecycleTracker::destructions = 0;

TEST(AlignedAllocatorTest, Alignment) {
    const size_t Alignment = 64; // Test with a large alignment
    AlignedAllocator<double, Alignment> allocator;
    
    // Allocate 10 doubles
    double* p = allocator.allocate(10);
    ASSERT_NE(p, nullptr);
    
    // Check alignment: pointer address must be divisible by Alignment
    uintptr_t address = reinterpret_cast<uintptr_t>(p);
    EXPECT_EQ(address % Alignment, 0u) << "Memory address " << address << " is not aligned to " << Alignment;
    
    allocator.deallocate(p, 10);
}

TEST(AlignedAllocatorTest, Lifecycle) {
    LifecycleTracker::constructions = 0;
    LifecycleTracker::destructions = 0;
    
    AlignedAllocator<LifecycleTracker, 32> allocator;
    LifecycleTracker* p = allocator.allocate(1);
    
    allocator.construct(p, LifecycleTracker());
    EXPECT_EQ(LifecycleTracker::constructions, 2); // 1 for temp, 1 for copy-construct into p
    
    allocator.destroy(p);
    EXPECT_EQ(LifecycleTracker::destructions, 2); // 1 for temp, 1 for p
    
    allocator.deallocate(p, 1);
}

TEST(AlignedAllocatorTest, VectorCompatibility) {
    // This is the most important test for our project as we use it with std::vector
    const size_t Alignment = 32;
    std::vector<double, AlignedAllocator<double, Alignment>> vec;
    
    vec.reserve(100);
    vec.push_back(1.0);
    vec.push_back(2.0);
    
    uintptr_t address = reinterpret_cast<uintptr_t>(vec.data());
    EXPECT_EQ(address % Alignment, 0u);
    EXPECT_EQ(vec[0], 1.0);
    EXPECT_EQ(vec[1], 2.0);
}

TEST(AlignedAllocatorTest, MaxSize) {
    AlignedAllocator<int, 16> allocator;
    EXPECT_GT(allocator.max_size(), 0u);
    EXPECT_LT(allocator.max_size(), (size_t)-1);
}

TEST(AlignedAllocatorTest, Comparison) {
    AlignedAllocator<int, 16> a1;
    AlignedAllocator<double, 16> a2;
    AlignedAllocator<int, 32> a3;
    
    EXPECT_TRUE(a1 == a2); // Same alignment
    EXPECT_FALSE(a1 != a2);
    
    EXPECT_FALSE(a1 == a3); // Different alignment
    EXPECT_TRUE(a1 != a3);
}

TEST(AlignedAllocatorTest, Rebind) {
    using Alloc16 = AlignedAllocator<int, 16>;
    using Rebound = Alloc16::rebind<double>::other;
    
    bool is_same = std::is_same<Rebound, AlignedAllocator<double, 16>>::value;
    EXPECT_TRUE(is_same);
}

TEST(AlignedVectorTest, ResizeAndZeroCorrectness) {
    AlignedVector<double, 32> vec;
    
    // Test initial resize and zero
    vec.resize_and_zero(10);
    EXPECT_EQ(vec.size(), 10);
    for (size_t i = 0; i < 10; ++i)
    {
        EXPECT_EQ(vec[i], 0.0);
    }
    
    // Set some non-zero values
    vec[0] = 5.0;
    vec[5] = 12.0;
    
    // Test resize and zero with SAME size (fill path)
    vec.resize_and_zero(10);
    EXPECT_EQ(vec.size(), 10);
    for (size_t i = 0; i < 10; ++i)
    {
        EXPECT_EQ(vec[i], 0.0);
    }

    // Set some non-zero values again
    vec[3] = 4.0;

    // Test resize and zero with DIFFERENT size (reallocation path)
    vec.resize_and_zero(15);
    EXPECT_EQ(vec.size(), 15);
    for (size_t i = 0; i < 15; ++i)
    {
        EXPECT_EQ(vec[i], 0.0);
    }
}

