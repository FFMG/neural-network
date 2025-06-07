#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

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

  TaskQueue(const TaskQueue&) = delete;
  TaskQueue& operator=(const TaskQueue&) = delete;

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

template <typename R>
class TaskQueuePool
{
public:
  TaskQueuePool(int number_of_thread = 0) :
    _number_of_threads(number_of_thread),
    _threads_index(0)
  {
    if(number_of_thread == 0)
    {
      _number_of_threads = std::thread::hardware_concurrency() -1;
    }
  }

  ~TaskQueuePool()
  {
    clean();
  }

  void start() 
  {
    clean();
    _task_queues.reserve(_number_of_threads);
    for(int i = 0; i < _number_of_threads; ++i)
    {
      auto task_queue = new TaskQueue<R>();
      _task_queues.emplace_back(task_queue);
      task_queue->start();
    }
    std::cout << "ThreadPool initialized with " << _number_of_threads << " worker threads." << std::endl;    
  }

  void stop() 
  {
    for(auto& task_queue : _task_queues)
    {
      task_queue->stop();
    }
    std::cout << "ThreadPool stop." << std::endl;
  }

  inline int total_tasks()
  {
    int total_num_tasks = 0;
    for(auto& task_queue : _task_queues)
    {
      total_num_tasks += task_queue->total_tasks();
    }
    return total_num_tasks;
  }

  inline double average()
  {
    double average = 0.0;
    for(auto& task_queue : _task_queues)
    {
      average += task_queue->average();
    }
    return _number_of_threads > 0 ? average / _number_of_threads : 0.0;
  }

  template <class F, class... Args>
  void enqueue(F&& f, Args&&... args) 
  {
    _task_queues[_threads_index++]->enqueue(std::forward<F>(f), std::forward<Args>(args)...);
    if(_threads_index >= _number_of_threads)
    {
      _threads_index = 0;
    }
  }

  std::vector<R> get() 
  {
    std::vector<R> output;
    for(auto& task_queue : _task_queues)
    {
      auto this_output = task_queue->get();
      output.insert(output.end(), this_output.begin(), this_output.end());
    }
    return output;
  }

private:
  int _number_of_threads;
  std::vector<TaskQueue<R>*> _task_queues;
  int _threads_index;

  void clean()
  {
    for(auto& task_queue : _task_queues)
    {
      delete task_queue;
    }
    _task_queues.clear();
  }
};
