#include <gtest/gtest.h>

// A simple empty test to verify that Google Test is linked and running correctly.
TEST(NeuralNetworkTest, EmptyTest) {
    EXPECT_TRUE(true);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
