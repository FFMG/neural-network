#pragma once
#include <iostream>   // Required for std::cout and std::endl
#include <string>     // Required for std::string
#include <chrono>     // Required for time operations (std::chrono)
#include <iomanip>    // Required for std::put_time and std::setfill/std::setw for formatting
#include <sstream>    // Required for std::stringstream to build the time string

class Logger
{
private:
    // ANSI escape codes for text colors. These codes work on most modern terminals
    // (Linux, macOS, and recent versions of Windows Terminal/PowerShell).
    // They instruct the terminal to change the color of subsequent text.
    static constexpr const char* LogColorReset = "\033[0m";   // Resets text color to default
    static constexpr const char* LogColorRed   = "\033[31m";
    static constexpr const char* LogColorGreen = "\033[32m";
    static constexpr const char* LogColorYellow = "\033[33m";
    static constexpr const char* LogColorBlue   = "\033[34m";

public:
  enum class LogLevel 
  {
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

  template <typename... Args>
  void log_debug(Args&&... args) 
  {
    log(LogLevel::Debug, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void log_info(Args&&... args) const
  {
    log(LogLevel::Information, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void log_warning(Args&&... args) const
  {
    log(LogLevel::Warning, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void log_error(Args&&... args) const
  {
    log(LogLevel::Error, std::forward<Args>(args)...);
  }
private:
  LogLevel _min_log_level; // Stores the minimum logging level set by the user

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

  void print_args() const
  {
    // No-op: This function does nothing, serving as the stopping point
    // for the recursion in the print_args variadic template.
  }

  template <typename T, typename... Args>
  void print_args(T&& first_arg, Args&&... rest) const
  {
    std::cout << std::forward<T>(first_arg);
    // Recursively call print_args with the remaining arguments
    print_args(std::forward<Args>(rest)...);
  }

  template <typename... Args>
  void log(LogLevel level, Args&&... args) const
  {
    // Only log if the current message's level is at or above the minimum configured level
    if (level < _min_log_level) 
    {
        return;
    }

    // 1. Output the current time
    std::cout << get_current_time_string() << " ";

    // 2. Determine color and tag based on log level
    const char* color_code = LogColorReset; // Default to no color
    std::string tag;

    switch (level) 
    {
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
    std::cout << color_code << tag << LogColorReset << " ";

    // 4. Output the user's message arguments
    print_args(std::forward<Args>(args)...);

    // 5. End the line
    std::cout << std::endl;
  }
};