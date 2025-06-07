#pragma once
#include <thread>
#include <mutex>
#include <queue>
#include <vector>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>

template <typename R>
class TaskQueue 
{
public:
  TaskQueue() : 
    _running(false),
    _total_tasks_durations(0),
    _total_num_tasks(0)
  {
  }
  ~TaskQueue() 
  {
    stop();
  }

  void start() 
  {
    if (_running) 
    {
      return;
    }
    _running = true;
    _worker = std::thread([this] { this->run(); });
  }

  void stop() 
  {
    {
      std::unique_lock<std::mutex> lock(_mutext);
      _running = false;
    }
    _condition_variable.notify_all();
    if (_worker.joinable()) 
    {
      _worker.join();
    }
  }

  // Accepts any callable + arguments matching the return type R
  template <class F, class... Args>
  void enqueue(F&& f, Args&&... args) 
  {
    auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    {
      std::unique_lock<std::mutex> lock(_mutext);
      _tasks.emplace([task]() -> R { return task(); });
    }
    _condition_variable.notify_one();
  }

  std::vector<R> get() 
  {
    std::unique_lock<std::mutex> lock(_mutext);
    std::vector<R> output;
    output.swap(_results); // clear and return
    return output;
  }

  inline int total_tasks()
  {
    std::unique_lock<std::mutex> lock(_mutext);
    return _total_num_tasks;
  }

  inline double average()
  {
    std::unique_lock<std::mutex> lock(_mutext);
    return _total_num_tasks > 0 ? static_cast<double>(_total_tasks_durations) / _total_num_tasks : 0.0;
  }

private:
  std::atomic<bool> _running;
  std::thread _worker;
  std::mutex _mutext;
  std::condition_variable _condition_variable;
  std::chrono::nanoseconds::rep _total_tasks_durations;
  int _total_num_tasks;

  std::queue<std::function<R()>> _tasks;
  std::vector<R> _results;

  void run() 
  {
    while (true) 
    {
      std::function<R()> task;
      {
        std::unique_lock<std::mutex> lock(_mutext);
        _condition_variable.wait(lock, [this] 
          { 
            return !_tasks.empty() || !_running; }
          );
        if (!_running && _tasks.empty()) 
        {
          return;
        }
        task = std::move(_tasks.front());
        _tasks.pop();
      }
      auto start = std::chrono::steady_clock::now();
      R result = task();
      auto end = std::chrono::steady_clock::now();
      {
        std::unique_lock<std::mutex> lock(_mutext);
        _total_tasks_durations += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        ++_total_num_tasks;
        _results.push_back(std::move(result));
      }
    }
  }
};
