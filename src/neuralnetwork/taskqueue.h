#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "logger.h"
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
    _total_completed_tasks(0),
    _total_pending_tasks(0)
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
    State old_state = State::Started;
    if (_state.compare_exchange_strong(old_state, State::Stopping))
    {
      _condition_new_task.notify_all();
      _condition_busy_task_complete.notify_all();
    }

    if (_worker.joinable())
    {
      _worker.join();
    }
    _state.store(State::Stopped);
  }

  // Accepts any callable + arguments matching the return type R
  template <class F, class... Args>
  void enqueue(F&& f, Args&&... args) 
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    if (_state.load(std::memory_order_relaxed) != State::Started)
    {
      if (_state.load() != State::Stopped) 
      {
        throw std::runtime_error("Cannot enqueue to a stopping queue.");
      }
    }

    auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    {
      std::unique_lock<std::mutex> lock(_mutex);
      if(_state.load() == State::Stopping)
      {
        throw std::invalid_argument("The task queue is stopping!");
      }
      if(_state.load() == State::Stopped)
      {
        _worker = std::thread([this] { this->run(); });
        _state.store(State::Started);
      }
      _tasks.emplace([task]() -> R { return task(); });
    }

    // Increment pending tasks atomically and notify worker.
    _total_pending_tasks.fetch_add(1, std::memory_order_relaxed);
    _condition_new_task.notify_one();
  }

  std::vector<R> get()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    std::vector<R> output;
    std::unique_lock<std::mutex> lock(_mutex);
    _condition_busy_task_complete.wait(lock, [this] 
      {
        return _total_pending_tasks.load() == 0;
      });

    output.swap(_results);
    _total_completed_tasks.store(0);
    return output;
  }

  inline int total_tasks() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    return total_completed_tasks() + total_pending_tasks(); // there is a small risk of a race condition
  }

  inline int total_completed_tasks() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    return _total_completed_tasks.load();
  }

  inline int total_pending_tasks() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    return _total_pending_tasks.load();
  }

private:
  void wait_for_all_tasks(std::unique_lock<std::mutex>& lock)
  {
    _condition_busy_task_complete.wait(lock, [this] 
    {
      return _total_pending_tasks.load() == 0;
    });
  }

  void move_local_results_to_results(std::vector<R>& local_results)
  {
    // we are done, do we have anything to move.
    if (local_results.empty())
    {
      return;
    }
    std::unique_lock<std::mutex> lock(_mutex);
    _results.insert(_results.end(),
      std::make_move_iterator(local_results.begin()),
      std::make_move_iterator(local_results.end()));
    local_results.clear();
  }

  void run()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    std::vector<R> local_results;
    local_results.reserve(100); // pre-allocate some space.
    while (_state.load(std::memory_order_relaxed) == State::Started)
    {
      std::function<R()> task;
      {
        std::unique_lock<std::mutex> lock(_mutex);
        _condition_new_task.wait(lock, [this]
          {
            return !_tasks.empty() || _state.load() != State::Started;
          });
        if (_state.load() != State::Started && _tasks.empty())
        {
          return;
        }

        task = std::move(_tasks.front());
        _tasks.pop();
      }

      try
      {
        R result = task();
        {
          local_results.push_back(std::move(result));
          _total_completed_tasks.fetch_add(1, std::memory_order_relaxed);

          // If this was the last pending task, notify waiters.
          if (_total_pending_tasks.fetch_sub(1, std::memory_order_acq_rel) == 1)
          {
            move_local_results_to_results(local_results);
            _condition_busy_task_complete.notify_all();
          }
        }
      }
      catch(const std::exception& e)
      {
        Logger::error("TaskQueue caught exception in task:", e.what());

        // even after an error, it is complete.
        _total_completed_tasks.fetch_add(1, std::memory_order_relaxed);

        // If this was the last pending task, notify waiters.
        if (_total_pending_tasks.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
          move_local_results_to_results(local_results);
          _condition_busy_task_complete.notify_all();
        }
        throw; // re-throw after logging and notifying
      }
    }
    
    // final move of whatever might remain.
    move_local_results_to_results(local_results);
    _state.store(State::Stopped);
  }

  std::thread _worker;
  std::mutex _mutex;
  std::condition_variable _condition_busy_task_complete;
  std::condition_variable _condition_new_task;

  std::atomic<State> _state;
  std::atomic<int> _total_completed_tasks;
  std::atomic<int> _total_pending_tasks;

  std::queue<std::function<R()>> _tasks;
  std::vector<R> _results;
};

template <>
class TaskQueue<void>
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
    _total_completed_tasks(0),
    _total_pending_tasks(0)
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
    State old_state = State::Started;
    if (_state.compare_exchange_strong(old_state, State::Stopping))
    {
      _condition_new_task.notify_all();
      _condition_busy_task_complete.notify_all();
    }

    if (_worker.joinable())
    {
      _worker.join();
    }
    _state.store(State::Stopped);
  }

  // Accepts any callable + arguments matching the return type R
  template <class F, class... Args>
  void enqueue(F&& f, Args&&... args)
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    if (_state.load(std::memory_order_relaxed) != State::Started)
    {
      if (_state.load() != State::Stopped)
      {
        throw std::runtime_error("Cannot enqueue to a stopping queue.");
      }
    }

    auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    {
      std::unique_lock<std::mutex> lock(_mutex);
      if (_state.load() == State::Stopping)
      {
        throw std::invalid_argument("The task queue is stopping!");
      }
      if (_state.load() == State::Stopped)
      {
        _worker = std::thread([this] { this->run(); });
        _state.store(State::Started);
      }
      _tasks.emplace([task]() { task(); });
    }

    // Increment pending tasks atomically and notify worker.
    _total_pending_tasks.fetch_add(1, std::memory_order_relaxed);
    _condition_new_task.notify_one();
  }

  void get()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    std::unique_lock<std::mutex> lock(_mutex);
    _condition_busy_task_complete.wait(lock, [this]
      {
        return _total_pending_tasks.load() == 0;
      });

    _total_completed_tasks.store(0);
  }

  inline int total_tasks() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    return total_completed_tasks() + total_pending_tasks(); // there is a small risk of a race condition
  }

  inline int total_completed_tasks() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    return _total_completed_tasks.load();
  }

  inline int total_pending_tasks() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    return _total_pending_tasks.load();
  }

private:
  void wait_for_all_tasks(std::unique_lock<std::mutex>& lock)
  {
    _condition_busy_task_complete.wait(lock, [this]
      {
        return _total_pending_tasks.load() == 0;
      });
  }

  void run()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueue");
    while (_state.load(std::memory_order_relaxed) == State::Started)
    {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(_mutex);
        _condition_new_task.wait(lock, [this]
          {
            return !_tasks.empty() || _state.load() != State::Started;
          });
        if (_state.load() != State::Started && _tasks.empty())
        {
          return;
        }

        task = std::move(_tasks.front());
        _tasks.pop();
      }

      try
      {
        task();
        {
          _total_completed_tasks.fetch_add(1, std::memory_order_relaxed);

          // If this was the last pending task, notify waiters.
          if (_total_pending_tasks.fetch_sub(1, std::memory_order_acq_rel) == 1)
          {
            _condition_busy_task_complete.notify_all();
          }
        }
      }
      catch (const std::exception& e)
      {
        Logger::error("TaskQueue caught exception in task:", e.what());

        // even after an error, it is complete.
        _total_completed_tasks.fetch_add(1, std::memory_order_relaxed);

        // If this was the last pending task, notify waiters.
        if (_total_pending_tasks.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
          _condition_busy_task_complete.notify_all();
        }
        throw; // re-throw after logging and notifying
      }
    }

    _state.store(State::Stopped);
  }

  std::thread _worker;
  std::mutex _mutex;
  std::condition_variable _condition_busy_task_complete;
  std::condition_variable _condition_new_task;

  std::atomic<State> _state;
  std::atomic<int> _total_completed_tasks;
  std::atomic<int> _total_pending_tasks;

  std::queue<std::function<void()>> _tasks;
};

template <typename R>
class SingleTaskQueue
{
private:
  enum class State {
    Started,
    Stopping,
    Stopped
  };

public:
  SingleTaskQueue() :
    _state(State::Stopped),
    _busy_task(false),
    _has_result(false),
    _task_is_present(false),
    _task(nullptr),
    _result(R())
  {
    MYODDWEB_PROFILE_FUNCTION("SingleTaskQueue");
  }
  ~SingleTaskQueue()
  {
    MYODDWEB_PROFILE_FUNCTION("SingleTaskQueue");
    stop();
  }

  SingleTaskQueue(const SingleTaskQueue&) = delete;
  SingleTaskQueue& operator=(const SingleTaskQueue&) = delete;

  void stop()
  {
    MYODDWEB_PROFILE_FUNCTION("SingleTaskQueue");
    auto old_state = State::Started;
    
    // Atomically change state to Stopping. Only the first thread to call stop() will do the work.
    if (_state.compare_exchange_strong(old_state, State::Stopping))
    {
      // Wake up all waiting threads (both worker and getters)
      _condition_new_task.notify_all();
      _condition_busy_task_complete.notify_all();

      if (_worker.joinable())
      {
        _worker.join();
      }
    }
    else // If it was already stopping or stopped, just wait for the worker to finish.
    {
      if (_worker.joinable())
      {
        _worker.join();
      }
    }
    _state.store(State::Stopped);
  }

  // Accepts any callable + arguments matching the return type R
  template <class F, class... Args>
  bool call(F&& f, Args&&... args)
  {
    MYODDWEB_PROFILE_FUNCTION("SingleTaskQueue");
    if (_busy_task.load() || _task_is_present.load())
    {
      return false; // we already have a task, so we can't add another one.
    }
    auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    {
      std::unique_lock<std::mutex> lock(_mutex);
      if (_state.load() == State::Stopping)
      {
        throw std::invalid_argument("The task queue is stopping!");
      }
      if (_state.load() == State::Stopped)
      {
        _worker = std::thread([this] { this->run(); });
        _state.store(State::Started);
      }
      _task = [task]() -> R { return task(); };
      _task_is_present.store(true);
    }

    _condition_new_task.notify_one();
    return true;
  }

  R get()
  {
    MYODDWEB_PROFILE_FUNCTION("SingleTaskQueue");
    std::unique_lock<std::mutex> lock(_mutex);
    wait_for_task(lock);

    R output;
    output = _result; // clear and return
    _has_result.store(false);
    return output;
  }

  inline bool busy() const
  {
    return _busy_task.load() || _task_is_present.load();
  }

  inline bool has_result() const
  {
    return _has_result.load();
  }

private:
  void wait_for_task(std::unique_lock<std::mutex>& lock)
  {
    if (!busy())
    {
      return;
    }
    _condition_busy_task_complete.wait(lock, [this]
      {
        return !busy();
      });
  }

  void run()
  {
    MYODDWEB_PROFILE_FUNCTION("SingleTaskQueue");
    _state.store(State::Started);
    while (_state.load(std::memory_order_relaxed) == State::Started)
    {
      std::function<R()> task;
      {
        std::unique_lock<std::mutex> lock(_mutex);
        _condition_new_task.wait(lock, [this]
          {
            return _task_is_present.load() || _state.load() != State::Started;
          });
        if (_state.load() != State::Started && !_task_is_present.load())
        {
          return;
        }
        task = std::move(_task);
        _busy_task.store(true);
        _has_result.store(false);
        _task_is_present.store(false);
      }

      try
      {
        R result = task();
        {
          std::unique_lock<std::mutex> lock(_mutex);
          _result = std::move(result);
          _has_result.store(true);
          _busy_task.store(false);
          _condition_busy_task_complete.notify_all();
        }
      }
      catch(const std::exception& e)
      {
        std::unique_lock<std::mutex> lock(_mutex);
        _busy_task.store(false);
        _condition_busy_task_complete.notify_all();
        Logger::error("SingleTaskQueue caught exception in task:", e.what());
        throw; // re-throw after logging and notifying
      }
    }
    _state.store(State::Stopped);
  }

  std::thread _worker;
  std::mutex _mutex;
  std::condition_variable _condition_busy_task_complete;
  std::condition_variable _condition_new_task;

  std::atomic<State> _state;
  std::atomic<bool> _busy_task;
  std::atomic<bool> _has_result;
  std::atomic<bool> _task_is_present;

  std::function<R()> _task;
  R _result;
};

template <>
class SingleTaskQueue<void>
{
private:
  enum class State {
    Started,
    Stopping,
    Stopped
  };

public:
  SingleTaskQueue() :
    _state(State::Stopped),
    _busy_task(false),
    _task_is_present(false),
    _task(nullptr)
  {
    MYODDWEB_PROFILE_FUNCTION("SingleTaskQueue");
  }
  ~SingleTaskQueue()
  {
    MYODDWEB_PROFILE_FUNCTION("SingleTaskQueue");
    stop();
  }

  SingleTaskQueue(const SingleTaskQueue&) = delete;
  SingleTaskQueue& operator=(const SingleTaskQueue&) = delete;

  void stop()
  {
    MYODDWEB_PROFILE_FUNCTION("SingleTaskQueue");
    auto old_state = State::Started;

    // Atomically change state to Stopping. Only the first thread to call stop() will do the work.
    if (_state.compare_exchange_strong(old_state, State::Stopping))
    {
      // Wake up all waiting threads (both worker and getters)
      _condition_new_task.notify_all();
      _condition_busy_task_complete.notify_all();

      if (_worker.joinable())
      {
        _worker.join();
      }
    }
    else // If it was already stopping or stopped, just wait for the worker to finish.
    {
      if (_worker.joinable())
      {
        _worker.join();
      }
    }
    _state.store(State::Stopped);
  }

  // Accepts any callable + arguments matching the return type R
  template <class F, class... Args>
  bool call(F&& f, Args&&... args)
  {
    MYODDWEB_PROFILE_FUNCTION("SingleTaskQueue");
    if (_busy_task.load() || _task_is_present.load())
    {
      return false; // we already have a task, so we can't add another one.
    }
    auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    {
      std::unique_lock<std::mutex> lock(_mutex);
      if (_state.load() == State::Stopping)
      {
        throw std::invalid_argument("The task queue is stopping!");
      }
      if (_state.load() == State::Stopped)
      {
        _worker = std::thread([this] { this->run(); });
        _state.store(State::Started);
      }
      _task = [task]() { task(); };
      _task_is_present.store(true);
    }

    _condition_new_task.notify_one();
    return true;
  }

  void get()
  {
    MYODDWEB_PROFILE_FUNCTION("SingleTaskQueue");
    std::unique_lock<std::mutex> lock(_mutex);
    wait_for_task(lock);
  }

  inline bool busy() const
  {
    return _busy_task.load() || _task_is_present.load();
  }

  inline bool has_result() const
  {
    return false; // there are no results for void
  }

private:
  void wait_for_task(std::unique_lock<std::mutex>& lock)
  {
    if (!busy())
    {
      return;
    }
    _condition_busy_task_complete.wait(lock, [this]
      {
        return !busy();
      });
  }

  void run()
  {
    MYODDWEB_PROFILE_FUNCTION("SingleTaskQueue");
    _state.store(State::Started);
    while (_state.load(std::memory_order_relaxed) == State::Started)
    {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(_mutex);
        _condition_new_task.wait(lock, [this]
          {
            return _task_is_present.load() || _state.load() != State::Started;
          });
        if (_state.load() != State::Started && !_task_is_present.load())
        {
          return;
        }
        task = std::move(_task);
        _busy_task.store(true);
        _task_is_present.store(false);
      }

      try
      {
        task();
        {
          std::unique_lock<std::mutex> lock(_mutex);
          _busy_task.store(false);
          _condition_busy_task_complete.notify_all();
        }
      }
      catch (const std::exception& e)
      {
        std::unique_lock<std::mutex> lock(_mutex);
        _busy_task.store(false);
        _condition_busy_task_complete.notify_all();
        Logger::error("SingleTaskQueue caught exception in task:", e.what());
        throw; // re-throw after logging and notifying
      }
    }
    _state.store(State::Stopped);
  }

  std::thread _worker;
  std::mutex _mutex;
  std::condition_variable _condition_busy_task_complete;
  std::condition_variable _condition_new_task;

  std::atomic<State> _state;
  std::atomic<bool> _busy_task;
  std::atomic<bool> _task_is_present;

  std::function<void()> _task;
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
    if (number_of_thread > 0)
    {
      _number_of_threads = number_of_thread;
    }
    else
    {
      auto hardware_threads = std::thread::hardware_concurrency();
      _number_of_threads = (hardware_threads > 1) ? hardware_threads - 1 : 1;
      if (hardware_threads == 0)
      {
        _number_of_threads = 2;
      }
    }
    start();
  }

  ~TaskQueuePool()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    stop();
  }

  void stop() 
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    for(auto& task_queue : _task_queues)
    {
      task_queue->stop();
    }
    Logger::debug([=]
      {
        return "ThreadPool stop.";
      });
  }

  inline int total_tasks() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    int total_num_tasks = 0;
    for (auto& task_queue : _task_queues)
    {
      total_num_tasks += task_queue->total_completed_tasks();
      total_num_tasks += task_queue->total_pending_tasks();
    }
    return total_num_tasks;
  }

  inline int total_completed_tasks() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    int total_num_tasks = 0;
    for(auto& task_queue : _task_queues)
    {
      total_num_tasks += task_queue->total_completed_tasks();
    }
    return total_num_tasks;
  }

  inline int total_pending_tasks() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    int total_num_tasks = 0;
    for (auto& task_queue : _task_queues)
    {
      total_num_tasks += task_queue->total_pending_tasks();
    }
    return total_num_tasks;
  }

  template <class F, class... Args>
  void enqueue(F&& f, Args&&... args) 
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    unsigned int index = _threads_index.fetch_add(1, std::memory_order_relaxed) % _number_of_threads;
    _task_queues[index]->enqueue(std::forward<F>(f), std::forward<Args>(args)...);
  }

  std::vector<R> get()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    size_t expcted = 0;
    for (auto& task_queue : _task_queues)
    {
      expcted += task_queue->total_tasks();
    }

    std::vector<R> output;
    output.reserve(expcted);
    for (auto& task_queue : _task_queues)
    {
      auto part = task_queue->get();
      if (!part.empty())
      {
        output.insert(output.end(),
          std::make_move_iterator(part.begin()),
          std::make_move_iterator(part.end()));
      }
    }

    // reset the thread index so we don't start new threads for no reason.
    _threads_index.store(0, std::memory_order_relaxed);
    return output;
  }

private:
  unsigned int _number_of_threads;
  std::vector<std::unique_ptr<TaskQueue<R>>> _task_queues;
  std::atomic<unsigned int> _threads_index;

  void start() 
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    _task_queues.clear();
    _task_queues.reserve(_number_of_threads);
    for (unsigned int i = 0; i < _number_of_threads; ++i)
    {
      _task_queues.emplace_back(std::make_unique<TaskQueue<R>>());
    }
    Logger::info([&] {
      return Logger::factory("ThreadPool initialized with ", _number_of_threads, " worker threads.");
      });
  }
};

template <>
class TaskQueuePool<void>
{
public:
  TaskQueuePool(int number_of_thread = 0) :
    _number_of_threads(number_of_thread),
    _threads_index(0)
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    if (number_of_thread > 0)
    {
      _number_of_threads = number_of_thread;
    }
    else
    {
      auto hardware_threads = std::thread::hardware_concurrency();
      _number_of_threads = (hardware_threads > 1) ? hardware_threads - 1 : 1;
      if (hardware_threads == 0)
      {
        _number_of_threads = 2;
      }
    }
    start();
  }

  ~TaskQueuePool()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    stop();
  }

  void stop()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    for (auto& task_queue : _task_queues)
    {
      task_queue->stop();
    }
    Logger::debug([=]
      {
        return "ThreadPool stop.";
      });
  }

  inline int total_tasks() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    int total_num_tasks = 0;
    for (auto& task_queue : _task_queues)
    {
      total_num_tasks += task_queue->total_completed_tasks();
      total_num_tasks += task_queue->total_pending_tasks();
    }
    return total_num_tasks;
  }

  inline int total_completed_tasks() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    int total_num_tasks = 0;
    for (auto& task_queue : _task_queues)
    {
      total_num_tasks += task_queue->total_completed_tasks();
    }
    return total_num_tasks;
  }

  inline int total_pending_tasks() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    int total_num_tasks = 0;
    for (auto& task_queue : _task_queues)
    {
      total_num_tasks += task_queue->total_pending_tasks();
    }
    return total_num_tasks;
  }

  template <class F, class... Args>
  void enqueue(F&& f, Args&&... args)
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    unsigned int index = _threads_index.fetch_add(1, std::memory_order_relaxed) % _number_of_threads;
    _task_queues[index]->enqueue(std::forward<F>(f), std::forward<Args>(args)...);
  }

  void get()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    for (auto& task_queue : _task_queues)
    {
      task_queue->get();
    }

    // reset the thread index so we don't start new threads for no reason.
    _threads_index.store(0, std::memory_order_relaxed);
  }

private:
  unsigned int _number_of_threads;
  std::vector<std::unique_ptr<TaskQueue<void>>> _task_queues;
  std::atomic<unsigned int> _threads_index;

  void start()
  {
    MYODDWEB_PROFILE_FUNCTION("TaskQueuePool");
    _task_queues.clear();
    _task_queues.reserve(_number_of_threads);
    for (unsigned int i = 0; i < _number_of_threads; ++i)
    {
      _task_queues.emplace_back(std::make_unique<TaskQueue<void>>());
    }
    Logger::info([&] {
      return Logger::factory("ThreadPool initialized with ", _number_of_threads, " worker threads.");
      });
  }
};
