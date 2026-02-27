#pragma once

// C headers
#include <cassert>
#include <cctype>
#include <cstdio>
#ifndef NDEBUG
#include <cstring>
#endif

// C++ headers
#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#if __cplusplus >= 201703L
#include <string_view>
#endif
#include <type_traits>
#include <utility>

class Logger
{
public:
  enum class LogLevel
  {
    Trace,
    Debug,
    Information,
    Warning,
    Error,
    Panic,
    None
  };

  static std::string level_to_string(LogLevel level)
  {
    switch (level)
    {
    case LogLevel::Trace:        return "Trace";
    case LogLevel::Information:  return "Info";
    case LogLevel::Warning:      return "Warning";
    case LogLevel::Error:        return "Error";
    case LogLevel::Panic:        return "Panic";
    case LogLevel::None:         return "None";
    case LogLevel::Debug:
    default:                     return "Debug";
    }
  }

  static LogLevel string_to_level(const std::string& str)
  {
    std::string lower_str = str;
    // Convert the string to lowercase for case-insensitive comparison
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
      [](unsigned char c) { return std::tolower(c); });

    if (lower_str == "trace")
    {
      return LogLevel::Trace;
    }
    if (lower_str == "debug")
    {
      return LogLevel::Debug;
    }
    if (lower_str == "info" || lower_str == "information")
    {
      return LogLevel::Information;
    }
    if (lower_str == "warn" || lower_str == "warning")
    {
      return LogLevel::Warning;
    }
    if (lower_str == "error")
    {
      return LogLevel::Error;
    }
    if (lower_str == "panic")
    {
      return LogLevel::Panic;
    }
    if (lower_str == "none")
    {
      return LogLevel::None;
    }

    // If no match is found, throw an exception
    throw std::invalid_argument("Unknown log level: " + str);
  }

private:
  // ANSI escape codes for text colors. These codes work on most modern terminals
  // (Linux, macOS, and recent versions of Windows Terminal/PowerShell).
  // They instruct the terminal to change the color of subsequent text.
  static constexpr const char* LogColorReset = "\033[0m";   // Resets text color to default
  static constexpr const char* LogColorRed = "\033[31m";
  static constexpr const char* LogColorGreen = "\033[32m";
  static constexpr const char* LogColorYellow = "\033[33m";
  static constexpr const char* LogColorBlue = "\033[34m";
  static constexpr const char* LogColorCyan = "\033[36m";

  static constexpr const size_t TimeStringLen = 12; // "HH:MM:SS.mmm" is 12 characters long
  static constexpr const size_t TimeStringBufferSize = 16;
  static constexpr const size_t TagLen = 7; // e.g. "[trace]"

#if __cplusplus >= 201703L
  using MessageParam = std::string_view;
#else
  using MessageParam = std::string;
#endif

  Logger(LogLevel minLevel = LogLevel::Information) : _min_level(minLevel)
  {
  }

  // Static method to get the singleton instance
  static Logger& instance()
  {
    static Logger instance;
    return instance;
  }

public:
  ~Logger() = default;
  Logger(const Logger& src) = delete;
  Logger& operator=(const Logger& src) = delete;
  Logger(Logger&&) = delete;
  Logger& operator=(Logger&&) = delete;

  static void set_level(LogLevel level)
  {
    instance()._min_level = level;
  }

  static LogLevel get_level()
  {
    return instance()._min_level;
  }

  template <typename... Args>
  static void trace(Args&&... args)
  {
    log(LogLevel::Trace, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static void debug(Args&&... args)
  {
    log(LogLevel::Debug, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static void info(Args&&... args)
  {
    log(LogLevel::Information, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static void warning(Args&&... args)
  {
    log(LogLevel::Warning, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static void error(Args&&... args)
  {
    log(LogLevel::Error, std::forward<Args>(args)...);
  }

  template <typename... Args>
  [[noreturn]] static void panic(Args&&... args)
  {
    log(LogLevel::Panic, std::forward<Args>(args)...);
#if defined(_MSC_VER)
    __assume(0);
#elif defined(__clang__) || defined(__GNUC__)
    __builtin_unreachable();
#else
    ((void)0);
#endif
  }

  static inline bool can_trace()
  {
    return instance().can_log(LogLevel::Trace);
  }
  static inline bool can_debug()
  {
    return instance().can_log(LogLevel::Debug);
  }
  static inline bool can_info()
  {
    return instance().can_log(LogLevel::Information);
  }
  static inline bool can_warning()
  {
    return instance().can_log(LogLevel::Warning);
  }
  static inline bool can_error()
  {
    return instance().can_log(LogLevel::Error);
  }

  template <typename... Args>
  static std::string factory(Args&&... args)
  {
    std::ostringstream oss;
    print_args(oss, std::forward<Args>(args)...);
    return oss.str();
  }
private:
  LogLevel _min_level; // Stores the minimum logging level set by the user

  bool can_log(LogLevel level) const
  {
    return (level == LogLevel::Panic || level >= _min_level);
  }

  // Buffer for formatting user argument in log()
  static std::ostringstream& get_msg_oss()
  {
    thread_local std::ostringstream oss;
    oss.str("");
    oss.clear();
    return oss;
  }

  // Buffer for formatting the final log message with timestamp and tag
  static std::ostringstream& get_msg_fmt_oss()
  {
    thread_local std::ostringstream oss;
    oss.str("");
    oss.clear();
    return oss;
  }
  
  static void get_current_time_string(char (&buf)[TimeStringBufferSize])
  {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    struct tm local_tm;
#if defined(_MSC_VER)
    localtime_s(&local_tm, &in_time_t);
#else
    localtime_r(&in_time_t, &local_tm);
#endif

    snprintf(buf, TimeStringBufferSize, "%02d:%02d:%02d.%03d", 
      local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec, 
      static_cast<int>(millis.count()));
  }

  static void print_args(std::ostringstream&)
  {
    // No-op: This function does nothing, serving as the stopping point
    // for the recursion in the print_args variadic template.
  }

  // Recursive case: Appends the first argument to the string and processes the rest
  template <typename T, typename... Args>
  static void print_args(std::ostringstream& oss, T&& first_arg, Args&&... rest)
  {
    if constexpr (std::is_same_v<std::decay_t<T>, std::function<std::string()>>)
    {
      //  exact function
      oss << first_arg();
    }
    else if constexpr (std::is_invocable_r_v<std::string, std::decay_t<T>>)
    {
      //  lambda function
      oss << first_arg();
    }
    else
    {
      oss << std::forward<T>(first_arg); // Convert the first argument to string
    }

    // Recursively call print_args with the remaining arguments and append
    print_args(oss, std::forward<Args>(rest)...);
  }

  template <typename... Args>
  static void log(LogLevel level, Args&&... args)
  {
    if (!instance().can_log(level))
    {
      return;
    }
    auto& oss = get_msg_oss();
    print_args(oss, std::forward<Args>(args)...);
    oss << '\n';
#if __cplusplus >= 202002L
    instance().string(level, oss.view());
#elif __cplusplus >= 201703L
    {
      auto msg = oss.str();
      instance().string(level, msg);
    }
#else
    instance().string(level, oss.str());
#endif    
  }

  void string(LogLevel level, MessageParam message) const
  {
    // 1. The current time
    char time_str[TimeStringBufferSize];
    get_current_time_string(time_str);

    // 2. Determine color and tag based on log level
    const char* color_code = LogColorReset; // Default to no color
    const char* tag;

    switch (level)
    {
    case LogLevel::Trace:
      color_code = LogColorCyan;
      tag = "[trace]";
      break;

    case LogLevel::Debug:
      color_code = LogColorGreen;
      tag = "[debug]";
      break;

    case LogLevel::Information:
      color_code = LogColorBlue;
      tag = "[info ]";
      break;

    case LogLevel::Warning:
      color_code = LogColorYellow;
      tag = "[warn ]";
      break;

    case LogLevel::Error:
      color_code = LogColorRed;
      tag = "[error]";
      break;

    case LogLevel::Panic:
      color_code = LogColorRed;
      tag = "[panic]";
      break;

    case LogLevel::None: // Should not be reached for logging, but included for completeness
      return; // If somehow called with None, just return

      default:
        color_code = LogColorReset;
        tag = "[ unk ]";
        break;
    }

    // sanity check in case we add new tags
#ifndef NDEBUG
    assert(strlen(tag) == TagLen);
#endif
    // 3. prepare for output
    constexpr size_t indent_len = TimeStringLen + 1 + TagLen + 1; // time + space + tag + space
    char indent[24];
    std::fill_n(indent, indent_len, ' ');
    indent[indent_len] = '\0';

    // 4. Output the color-coded tag, then reset color
    auto& oss = get_msg_fmt_oss();
    oss << time_str << ' ' << color_code << tag << LogColorReset << ' ';

    // 5. Output the user's message arguments
    size_t start = 0;
    size_t pos = 0;
    while ((pos = message.find('\n', start)) != std::string::npos)
    {
      oss.write(message.data() + start, pos - start + 1);
      start = pos + 1;
      if(start < message.size())
      {
        oss.write(indent, indent_len);
      }
    }
    oss.write(message.data() + start, message.size() - start);

    // 6. Print the final message to the console
    std::cout << oss.str();

    if (level == LogLevel::Panic)
    {
      throw std::runtime_error(oss.str());
    }
  }
};