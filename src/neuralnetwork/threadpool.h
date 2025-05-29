#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

class ThreadPool 
{
public:
    ThreadPool();
    ThreadPool(size_t threads);
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    
    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> 
    {
      using return_type = std::invoke_result_t<F, Args...>; // C++17 way to get return type

      // Create a std::packaged_task to wrap the function and its arguments.
      // std::packaged_task allows us to get a std::future for the task's result.
      auto task = std::make_shared<std::packaged_task<return_type()>>(
          std::bind(std::forward<F>(f), std::forward<Args>(args)...)
      );

      // Get the future associated with this packaged_task
      std::future<return_type> res = task->get_future();
      { // SCOPE for unique_lock to protect the _queue_mutex
        std::unique_lock<std::mutex> lock(_queue_mutex);
        if (_stop) 
        {
          throw std::runtime_error("enqueue on stopped ThreadPool");
        }

        _tasks.emplace([task]() { (*task)(); });
      } // Mutex lock is released here

      // Notify one waiting worker thread that a new task is available
      _condition.notify_one();
      return res; // Return the future to the caller
    }

    virtual ~ThreadPool();

private:
    // The vector of worker threads
    std::vector<std::thread> _workers;

    // The queue of _tasks (each task is a std::function<void()>)
    std::queue<std::function<void()>> _tasks;

    // Mutex to protect access to the task queue
    std::mutex _queue_mutex;
    std::condition_variable _condition;

    bool _stop;
};