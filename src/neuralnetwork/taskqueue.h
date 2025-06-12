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

#include "./libraries/instrumentor.h"

template <typename R>
class TaskQueue
{
private:
  enum class State {
    Started,
    Stopping,
    Stopped
  };

public:
  TaskQueue() :
    _state(State::Stopped),
    _total_tasks_durations(0),
    _total_num_tasks(0),
    _busy_task(0)
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
  }
  ~TaskQueue()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    stop();
  }

  TaskQueue(const TaskQueue&) = delete;
  TaskQueue& operator=(const TaskQueue&) = delete;

  void stop() 
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    {
      std::unique_lock<std::mutex> lock(_mutext);
      _state = State::Stopping;
      if (_state != State::Stopped)
      {
        wait_for_all_tasks(lock);
      }
    }
    // tell everyone about this
    _condition_new_task.notify_all();
    if (_worker.joinable()) 
    {
      _worker.join();
    }
    _state = State::Stopped;
  }

  // Accepts any callable + arguments matching the return type R
  template <class F, class... Args>
  void enqueue(F&& f, Args&&... args) 
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    {
      std::unique_lock<std::mutex> lock(_mutext);
      if(_state == State::Stopping)
      {
        throw new std::invalid_argument("The task queue is stopping!");
      }
      if(_state == State::Stopped)
      {
        _worker = std::thread([this] { this->run(); });
        _state = State::Started;
      }
      _tasks.emplace([task]() -> R { return task(); });
    }
    // tell one thread about this.
    _condition_new_task.notify_one();
  }

  std::vector<R> get()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    std::unique_lock<std::mutex> lock(_mutext);
    wait_for_all_tasks(lock);

    std::vector<R> output;
    output.swap(_results); // clear and return
    return output;
  }

  inline int total_tasks()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    std::unique_lock<std::mutex> lock(_mutext);
    return _total_num_tasks;
  }

  inline double average()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    std::unique_lock<std::mutex> lock(_mutext);
    return _total_num_tasks > 0 ? static_cast<double>(_total_tasks_durations) / _total_num_tasks : 0.0;
  }

private:
  void wait_for_all_tasks(std::unique_lock<std::mutex>& lock)
  {
    if (_busy_task ==0 && _tasks.empty())
    {
      return;
    }
    _condition_busy_task_complete.wait(lock, [this] 
    {
      return _busy_task == 0 && _tasks.empty();
    });
  }

  void run() 
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    _state = State::Started;
    while (true)
    {
      std::function<R()> task;
      {
        std::unique_lock<std::mutex> lock(_mutext);
        _condition_new_task.wait(lock, [this]
          {
            return !_tasks.empty() || (_state == State::Stopped || _state == State::Stopping);
          });
        if ((_state == State::Stopped || _state == State::Stopping) && _tasks.empty())
        {
          return;
        }
        task = std::move(_tasks.front());
        ++_busy_task; // this task is now busy
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
        --_busy_task; //  this task is no longer busy.
        _condition_busy_task_complete.notify_all();
      }
    }
    _state = State::Stopped;
  }

  std::thread _worker;
  std::mutex _mutext;
  std::condition_variable _condition_busy_task_complete;
  std::condition_variable _condition_new_task;

  std::atomic<State> _state;
  std::chrono::nanoseconds::rep _total_tasks_durations;
  int _total_num_tasks;
  int _busy_task;

  std::queue<std::function<R()>> _tasks;
  std::vector<R> _results;
};

template <typename R>
class TaskQueuePool
{
public:
  TaskQueuePool(int number_of_thread = 0) :
    _number_of_threads(number_of_thread),
    _threads_index(0)
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    if(number_of_thread == 0)
    {
      _number_of_threads = std::thread::hardware_concurrency() -1;
    }
    start();
  }

  ~TaskQueuePool()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    clean();
  }

  void stop() 
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    for(auto& task_queue : _task_queues)
    {
      task_queue->stop();
    }
    std::cout << "ThreadPool stop." << std::endl;
  }

  inline int total_tasks()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    int total_num_tasks = 0;
    for(auto& task_queue : _task_queues)
    {
      total_num_tasks += task_queue->total_tasks();
    }
    return total_num_tasks;
  }

  inline double average()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    double average = 0.0;
    int used_averages = 0;
    for(auto& task_queue : _task_queues)
    {
      if (task_queue->total_tasks() == 0)
      {
        continue;
      }
      ++used_averages;
      average += task_queue->average();
    }
    return used_averages > 0 ? average / used_averages : 0.0;
  }

  template <class F, class... Args>
  void enqueue(F&& f, Args&&... args) 
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    _task_queues[_threads_index++]->enqueue(std::forward<F>(f), std::forward<Args>(args)...);
    if(_threads_index >= _number_of_threads)
    {
      _threads_index = 0;
    }
  }

  std::vector<R> get()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    std::vector<R> output;
    for(auto& task_queue : _task_queues)
    {
      auto this_output = task_queue->get();
      output.insert(output.end(), this_output.begin(), this_output.end());
    }
    // reset the thread index so we don't start new threads for no reason.
    _threads_index = 0;
    return output;
  }

private:
  int _number_of_threads;
  std::vector<TaskQueue<R>*> _task_queues;
  int _threads_index;

  void clean()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    for(auto& task_queue : _task_queues)
    {
      delete task_queue;
    }
    _task_queues.clear();
  }

  void start() 
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    clean();
    _task_queues.reserve(_number_of_threads);
    for(int i = 0; i < _number_of_threads; ++i)
    {
      auto task_queue = new TaskQueue<R>();
      _task_queues.emplace_back(task_queue);
    }
    std::cout << "ThreadPool initialized with " << _number_of_threads << " worker threads." << std::endl;    
  }  
};
