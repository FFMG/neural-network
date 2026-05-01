#include <gtest/gtest.h>
#include "..\..\src\neuralnetwork\logger.h"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    Logger::set_level(Logger::LogLevel::Information);
    return RUN_ALL_TESTS();
}
