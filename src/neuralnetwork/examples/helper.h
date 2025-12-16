#include <iostream>
#include <chrono>
#include <string>

#define TEST_START(label) \
    auto __test_start_time = std::chrono::high_resolution_clock::now(); \
    const std::string __test_label = label;

#define TEST_END \
    do { \
        auto __test_end_time = std::chrono::high_resolution_clock::now(); \
        auto __test_duration = std::chrono::duration_cast<std::chrono::milliseconds>(__test_end_time - __test_start_time).count(); \
        long long __test_minutes = __test_duration / 60000; \
        long long __test_seconds = (__test_duration % 60000) / 1000; \
        long long __test_millis  = __test_duration % 1000; \
        Logger::info( __test_label, " took " \
                  , __test_minutes ," min " \
                  , __test_seconds ," sec " \
                  , __test_millis  ," ms\n"); \
    } while(0);
