#include <gtest/gtest.h>
#include "common/logger.h"

#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace myoddweb::nn;

namespace
{
// RAII helper: redirects std::cout to the given buffer for the lifetime of
// the object, and always restores the original buffer, even if the body
// throws.
class CoutRedirect
{
public:
  explicit CoutRedirect(std::streambuf* new_buffer) :
    _old_buffer(std::cout.rdbuf(new_buffer))
  {
  }

  ~CoutRedirect()
  {
    std::cout.rdbuf(_old_buffer);
  }

  CoutRedirect(const CoutRedirect&) = delete;
  CoutRedirect& operator=(const CoutRedirect&) = delete;

private:
  std::streambuf* _old_buffer;
};

// A stringbuf that records how many times sync() (i.e. a flush) was
// requested, so tests can verify Logger actually flushes when it must.
class SyncCountingBuffer : public std::stringbuf
{
public:
  int sync_count = 0;

protected:
  int sync() override
  {
    ++sync_count;
    return std::stringbuf::sync();
  }
};

struct LogWorker
{
  int thread_id;
  int messages_per_thread;
  size_t message_length;

  void operator()() const
  {
    const std::string payload(message_length, static_cast<char>('A' + thread_id));
    for (int i = 0; i < messages_per_thread; ++i)
    {
      Logger::error(payload);
    }
  }
};

struct LazyCounter
{
  int* call_count;

  std::string operator()() const
  {
    ++(*call_count);
    return "expensive debug info";
  }
};

struct LevelReader
{
  std::atomic<bool>* running;
  std::atomic<int>* total_reads;

  void operator()() const
  {
    while (running->load(std::memory_order_relaxed))
    {
      auto lvl = Logger::get_level();
      (void)lvl;
      (void)Logger::can_trace();
      (void)Logger::can_info();
      (void)Logger::can_error();
      total_reads->fetch_add(1, std::memory_order_relaxed);
    }
  }
};

struct LevelWriter
{
  std::atomic<bool>* running;

  void operator()() const
  {
    const Logger::LogLevel levels[] =
    {
      Logger::LogLevel::Trace,
      Logger::LogLevel::Debug,
      Logger::LogLevel::Information,
      Logger::LogLevel::Warning,
      Logger::LogLevel::Error
    };
    int i = 0;
    while (running->load(std::memory_order_relaxed))
    {
      Logger::set_level(levels[i % 5]);
      ++i;
    }
  }
};

class LoggerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    _original_level = Logger::get_level();
  }

  void TearDown() override
  {
    Logger::set_level(_original_level);
  }

private:
  Logger::LogLevel _original_level = Logger::LogLevel::Information;
};
} // namespace

TEST_F(LoggerTest, MinimumLevelFiltersLowerSeverityMessages)
{
  Logger::set_level(Logger::LogLevel::Warning);
  EXPECT_FALSE(Logger::can_trace());
  EXPECT_FALSE(Logger::can_debug());
  EXPECT_FALSE(Logger::can_info());
  EXPECT_TRUE(Logger::can_warning());
  EXPECT_TRUE(Logger::can_error());
}

TEST_F(LoggerTest, PanicAlwaysThrowsEvenWhenLoggingIsFullyDisabled)
{
  // LogLevel::None is meant to suppress every level, but panic() must still
  // fire: it is the caller's only signal that something fatal happened.
  Logger::set_level(Logger::LogLevel::None);

  std::ostringstream capture;
  CoutRedirect redirect(capture.rdbuf());

  bool threw = false;
  try
  {
    Logger::panic("catastrophic failure: ", 42);
  }
  catch (const std::runtime_error& e)
  {
    threw = true;
    EXPECT_NE(std::string(e.what()).find("catastrophic failure: 42"), std::string::npos);
  }
  EXPECT_TRUE(threw);
}

TEST_F(LoggerTest, WarningErrorAndPanicFlushImmediately)
{
  // std::cout is buffered. If the process later dies abnormally (for
  // example an uncaught exception escaping a std::thread, which calls
  // std::terminate/abort without unwinding or flushing streams), anything
  // still sitting in the buffer is lost. Warning/Error/Panic must be
  // flushed the moment they are written so the message always survives.
  Logger::set_level(Logger::LogLevel::Trace);

  SyncCountingBuffer buffer;
  {
    CoutRedirect redirect(&buffer);
    Logger::warning("warning message");
    Logger::error("error message");
    EXPECT_THROW(Logger::panic("panic message"), std::runtime_error);
  }

  EXPECT_EQ(buffer.sync_count, 3);
}

TEST_F(LoggerTest, TraceDebugAndInfoDoNotFlushImmediately)
{
  // Low-severity, high-frequency levels are left buffered for performance;
  // only Warning and above pay the cost of an explicit flush.
  Logger::set_level(Logger::LogLevel::Trace);

  SyncCountingBuffer buffer;
  {
    CoutRedirect redirect(&buffer);
    Logger::trace("trace message");
    Logger::debug("debug message");
    Logger::info("info message");
  }

  EXPECT_EQ(buffer.sync_count, 0);
}

TEST_F(LoggerTest, ConcurrentLoggingDoesNotInterleaveMessages)
{
  // Logger has no per-call synchronisation of its own guaranteed by the
  // language for std::cout: concurrent writers can otherwise interleave
  // characters mid-line. Every line captured below must be made up of a
  // single thread's repeated marker character, never a mix of two.
  Logger::set_level(Logger::LogLevel::Information);

  constexpr int thread_count = 8;
  constexpr int messages_per_thread = 25;
  constexpr size_t message_length = 40;

  std::ostringstream capture;
  {
    CoutRedirect redirect(capture.rdbuf());

    std::vector<std::thread> threads;
    for (int t = 0; t < thread_count; ++t)
    {
      threads.emplace_back(LogWorker{ t, messages_per_thread, message_length });
    }
    for (auto& thread : threads)
    {
      thread.join();
    }
  }

  std::istringstream lines(capture.str());
  std::string line;
  size_t checked_lines = 0;
  while (std::getline(lines, line))
  {
    if (line.size() < message_length)
    {
      continue;
    }
    const std::string tail = line.substr(line.size() - message_length);
    const char expected = tail.front();
    for (char c : tail)
    {
      EXPECT_EQ(c, expected) << "Interleaved output detected: " << line;
    }
    ++checked_lines;
  }
  EXPECT_EQ(checked_lines, static_cast<size_t>(thread_count * messages_per_thread));
}

TEST_F(LoggerTest, LevelStringRoundTripIsCaseInsensitive)
{
  EXPECT_EQ(Logger::string_to_level("WARNING"), Logger::LogLevel::Warning);
  EXPECT_EQ(Logger::string_to_level("warn"), Logger::LogLevel::Warning);
  EXPECT_EQ(Logger::level_to_string(Logger::LogLevel::Error), "Error");
  EXPECT_THROW(Logger::string_to_level("not-a-level"), std::invalid_argument);
}

TEST_F(LoggerTest, MultiLineMessageIndentsSubsequentLines)
{
  Logger::set_level(Logger::LogLevel::Information);

  std::ostringstream capture;
  {
    CoutRedirect redirect(capture.rdbuf());
    Logger::info("First line\nSecond line\nThird line");
  }

  std::istringstream lines(capture.str());
  std::string line1;
  std::string line2;
  std::string line3;

  ASSERT_TRUE(std::getline(lines, line1));
  ASSERT_TRUE(std::getline(lines, line2));
  ASSERT_TRUE(std::getline(lines, line3));

  EXPECT_NE(line1.find("First line"), std::string::npos);

  const std::string expected_indent(21, ' ');
  EXPECT_EQ(line2.rfind(expected_indent + "Second line", 0), 0u);
  EXPECT_EQ(line3.rfind(expected_indent + "Third line", 0), 0u);
}

TEST_F(LoggerTest, LazyLoggingCallableOnlyEvaluatedWhenSeverityEnabled)
{
  Logger::set_level(Logger::LogLevel::Warning);

  int call_count = 0;
  LazyCounter counter{ &call_count };

  Logger::trace(counter);
  Logger::debug(counter);
  EXPECT_EQ(call_count, 0);

  std::ostringstream capture;
  {
    CoutRedirect redirect(capture.rdbuf());
    Logger::warning(counter);
  }
  EXPECT_EQ(call_count, 1);
  EXPECT_NE(capture.str().find("expensive debug info"), std::string::npos);
}

TEST_F(LoggerTest, FactoryFormatsMixedTypesAndVectors)
{
  const std::vector<int> numbers = { 1, 2, 3 };
  const auto formatted = Logger::factory("Iteration: ", 42, ", loss: ", 0.05, ", items: ", numbers);
  EXPECT_EQ(formatted, "Iteration: 42, loss: 0.05, items: [1, 2, 3]");
}

TEST_F(LoggerTest, AllLogLevelsCanBeSetAndRetrieved)
{
  const std::vector<Logger::LogLevel> all_levels =
  {
    Logger::LogLevel::Trace,
    Logger::LogLevel::Debug,
    Logger::LogLevel::Information,
    Logger::LogLevel::Warning,
    Logger::LogLevel::Error,
    Logger::LogLevel::Panic,
    Logger::LogLevel::None
  };

  for (const auto level : all_levels)
  {
    Logger::set_level(level);
    EXPECT_EQ(Logger::get_level(), level);
  }
}

TEST_F(LoggerTest, AtomicLevelConcurrentReadAndWriteStress)
{
  std::atomic<bool> running = true;
  std::atomic<int> total_reads = 0;

  std::vector<std::thread> readers;
  for (int i = 0; i < 4; ++i)
  {
    readers.emplace_back(LevelReader{ &running, &total_reads });
  }

  std::vector<std::thread> writers;
  for (int i = 0; i < 2; ++i)
  {
    writers.emplace_back(LevelWriter{ &running });
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  running = false;

  for (auto& w : writers)
  {
    w.join();
  }
  for (auto& r : readers)
  {
    r.join();
  }

  EXPECT_GT(total_reads.load(), 0);
}
