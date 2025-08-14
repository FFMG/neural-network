#pragma once
#include <chrono>     // Required for time operations (std::chrono)
#include <functional>
#include <iomanip>    // Required for std::put_time and std::setfill/std::setw for formatting
#include <iostream>   // Required for std::cout and std::endl
#include <string>     // Required for std::string
#include <sstream>    // Required for std::stringstream to build the time string

class Logger
{
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

  Logger(LogLevel minLevel = LogLevel::Information) : _min_log_level(minLevel) 
  {

  }
  ~Logger() = default;

  Logger(const Logger& src) : 
    _min_log_level(src._min_log_level)
  {
    
  }
  Logger& operator=(const Logger& src)
  {
    if(this != &src)
    {
      _min_log_level = src._min_log_level;
    }
    return *this;
  }

  Logger(Logger&&) = delete;
  Logger& operator=(Logger&&) = delete;

  bool log(LogLevel level) const
  {
    // Only log if the current message's level is at or above the minimum configured level
    return level >= _min_log_level;
  }

  void log_tracef(std::function<std::string()> message_factory) const
  {
    log_with_factory(LogLevel::Trace, message_factory);
  }

  template <typename... Args>
  void log_trace(Args&&... args) const
  {
    log(LogLevel::Trace, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void log_debug(Args&&... args) const
  {
    log(LogLevel::Debug, std::forward<Args>(args)...);
  }

  void log_debugf(std::function<std::string()> message_factory) const
  {
    log_with_factory(LogLevel::Debug, message_factory);
  }

  template <typename... Args>
  void log_info(Args&&... args) const
  {
    log(LogLevel::Information, std::forward<Args>(args)...);
  }

  void log_infof(std::function<std::string()> message_factory) const
  {
    log_with_factory(LogLevel::Information, message_factory);
  }

  template <typename... Args>
  void log_warning(Args&&... args) const
  {
    log(LogLevel::Warning, std::forward<Args>(args)...);
  }

  void log_warningf(std::function<std::string()> message_factory) const
  {
    log_with_factory(LogLevel::Warning, message_factory);
  }

  template <typename... Args>
  void log_error(Args&&... args) const
  {
    log(LogLevel::Error, std::forward<Args>(args)...);
  }

  void log_errorf(std::function<std::string()> message_factory) const
  {
    log_with_factory(LogLevel::Error, message_factory);
  }

  inline bool can_log_trace() const
  {
    return can_log(LogLevel::Debug);
  }
  inline bool can_log_debug() const
  {
    return can_log(LogLevel::Debug);
  }
  inline bool can_log_info() const
  {
    return can_log(LogLevel::Information);
  }
  inline bool can_log_warning() const
  {
    return can_log(LogLevel::Warning);
  }
  inline bool can_log_error() const
  {
    return can_log(LogLevel::Error);
  }

  template <typename... Args>
  static std::string log_factory(Args&&... args)
  {
    std::ostringstream oss;
    oss << print_args(std::forward<Args>(args)...) << std::endl;
    return  oss.str();
  }
private:
  LogLevel _min_log_level; // Stores the minimum logging level set by the user

  bool can_log(LogLevel level) const
  {
    // Only log if the current message's level is at or above the minimum configured level
    return level >= _min_log_level;
  }

  std::string get_current_time_string() const
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
    oss << std::forward<T>(first_arg); // Convert the first argument to string

    // Recursively call print_args with the remaining arguments and append
    oss << print_args(std::forward<Args>(rest)...);

    return oss.str();
  }

  void log_with_factory(LogLevel level, std::function<std::string()> message_factory) const
  {
    if (level < _min_log_level)
    {
      return;
    }

    // The function is only called if the log level is sufficient.
    std::ostringstream oss; 
    oss << message_factory();
    log_string(level, oss.str());
  }

  template <typename... Args>
  void log(LogLevel level, Args&&... args) const
  {
    std::ostringstream oss;
    oss << print_args(std::forward<Args>(args)...) << std::endl;
    log_string(level, oss.str());
  }

  void log_string(LogLevel level, std::string message) const
  {
    // Only log if the current message's level is at or above the minimum configured level
    if (level < _min_log_level)
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