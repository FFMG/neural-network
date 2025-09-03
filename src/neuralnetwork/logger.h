#pragma once
#include <algorithm>
#include <chrono>     // Required for time operations (std::chrono)
#include <functional>
#include <iomanip>    // Required for std::put_time and std::setfill/std::setw for formatting
#include <iostream>   // Required for std::cout and std::endl
#include <string>     // Required for std::string
#include <sstream>    // Required for std::stringstream to build the time string
#include <type_traits>

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

    if(lower_str == "trace")
    {
      return LogLevel::Trace;
    }
    if(lower_str == "debug")
    {
      return LogLevel::Debug;
    }      
    if(lower_str == "info" || lower_str == "information")
    {
      return LogLevel::Information;
    }
    if(lower_str == "warn" || lower_str == "warning") 
    {
      return LogLevel::Warning;
    }
    if(lower_str == "error")  
    {
      return LogLevel::Error;
    }
    if(lower_str == "none")  
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
  static constexpr const char* LogColorReset  = "\033[0m";   // Resets text color to default
  static constexpr const char* LogColorRed    = "\033[31m";
  static constexpr const char* LogColorGreen  = "\033[32m";
  static constexpr const char* LogColorYellow = "\033[33m";
  static constexpr const char* LogColorBlue   = "\033[34m";
  static constexpr const char* LogColorCyan   = "\033[36m";

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

  static inline bool can_trace()
  {
    return can_log(LogLevel::Trace);
  }
  static inline bool can_debug()
  {
    return can_log(LogLevel::Debug);
  }
  static inline bool can_info()
  {
    return can_log(LogLevel::Information);
  }
  static inline bool can_warning()
  {
    return can_log(LogLevel::Warning);
  }
  static inline bool can_error()
  {
    return can_log(LogLevel::Error);
  }

  template <typename... Args>
  static std::string factory(Args&&... args)
  {
    std::ostringstream oss;
    oss << print_args(std::forward<Args>(args)...) << std::endl;
    return  oss.str();
  }
private:
  LogLevel _min_level; // Stores the minimum logging level set by the user

  static bool can_log(LogLevel level)
  {
    // Only log if the current message's level is at or above the minimum configured level
    return level >= instance()._min_level;
  }

  static std::string get_current_time_string()
  {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << millis.count();
    return ss.str();
  }

  static std::string print_args()
  {
    // No-op: This function does nothing, serving as the stopping point
    // for the recursion in the print_args variadic template.
    return "";
  }

  // Recursive case: Appends the first argument to the string and processes the rest
  template <typename T, typename... Args>
  static std::string print_args(T&& first_arg, Args&&... rest)
  {
    std::ostringstream oss;

    if constexpr (std::is_same_v<std::decay_t<T>, std::function<std::string()>> )
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
    oss << print_args(std::forward<Args>(rest)...);

    return oss.str();
  }

  static void with_factory(LogLevel level, std::function<std::string()> message_factory)
  {
    if (level < instance()._min_level)
    {
      return;
    }

    // The function is only called if the log level is sufficient.
    std::ostringstream oss; 
    oss << message_factory();
    instance().string(level, oss.str());
  }

  template <typename... Args>
  static void log(LogLevel level, Args&&... args)
  {
    std::ostringstream oss;
    oss << print_args(std::forward<Args>(args)...) << std::endl;
    instance().string(level, oss.str());
  }

  void string(LogLevel level, std::string message) const
  {
    // Only log if the current message's level is at or above the minimum configured level
    if (level < _min_level)
    {
      return;
    }

    std::ostringstream oss;

    // 1. Output the current time
    oss << get_current_time_string() << " ";

    // 2. Determine color and tag based on log level
    const char* color_code = LogColorReset; // Default to no color
    std::string tag;

    switch (level) 
    {
    case LogLevel::Trace:
      color_code = LogColorCyan;
      tag = "[TRC]";
      break;

    case LogLevel::Debug:
      color_code = LogColorGreen;
      tag = "[DBG]";
      break;

    case LogLevel::Information:
      color_code = LogColorBlue;
      tag = "[INF]";
      break;

    case LogLevel::Warning:
        color_code = LogColorYellow;
        tag = "[WRN]";
        break;

    case LogLevel::Error:
        color_code = LogColorRed;
        tag = "[ERR]";
        break;

    case LogLevel::None: // Should not be reached for logging, but included for completeness
        return; // If somehow called with None, just return
    }

    // 3. Output the color-coded tag, then reset color
    oss << color_code << tag << LogColorReset << " ";

    // 4. Output the user's message arguments
    oss << message;

    // 5. Print the final message to the console
    std::cout << oss.str();
  }
};