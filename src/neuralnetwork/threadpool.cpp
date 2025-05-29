#include "threadpool.h"

#include <iostream>

ThreadPool::ThreadPool() :
  ThreadPool( 2* std::thread::hardware_concurrency() )
{
}

ThreadPool::ThreadPool(size_t threads) :
  _stop(false) 
{
  if (threads == 0) 
  {
    // Fallback for systems that might report 0 or if user passes 0
    threads = 1;
    std::cerr << "Warning: ThreadPool initialized with 0 threads. Setting to 1.\n";
  }

  // Create the worker threads
  for (size_t i = 0; i < threads; ++i) 
  {
    _workers.emplace_back(
      [this] { // Lambda function that each worker thread executes
        for (;;) 
        { // Infinite loop for worker threads
          std::function<void()> task; // Placeholder for the task to be executed
          { // SCOPE for unique_lock to protect the queue_mutex
            std::unique_lock<std::mutex> lock(_queue_mutex);

            // Wait until there's a task in the queue OR the pool is stopped
            // The lambda predicate is a safeguard against spurious wakeups
            _condition.wait(lock, [this]{
                return _stop || !_tasks.empty();
            });

            // If the pool is stopped AND the queue is empty, this worker thread can exit
            if (_stop && _tasks.empty())
                return;

            // Get the task from the front of the queue
            task = std::move(_tasks.front());
            _tasks.pop();
          } // Mutex lock is released here, allowing other threads to access the queue

          // Execute the task outside the mutex lock to avoid holding the lock
          // during long-running computations.
          try 
          {
            task();
          } catch (const std::exception& e) 
          {
            std::cerr << "ThreadPool task threw an exception: " << e.what() << std::endl;
            // Depending on your error handling needs, you might log this,
            // or rethrow if packaged_task handles it (which it does).
          } 
          catch (...) 
          {
            std::cerr << "ThreadPool task threw an unknown exception." << std::endl;
          }
        }
      }
    );
  }
}

ThreadPool::~ThreadPool() 
{
  { // SCOPE for unique_lock to protect the queue_mutex
    std::unique_lock<std::mutex> lock(_queue_mutex);
    _stop = true; // Set the stop flag to signal workers to exit
  } // Mutex lock is released here

  // Notify all waiting threads so they can check the 'stop' flag and exit
  _condition.notify_all();

  // Join all worker threads to ensure they finish their current task
  // and terminate gracefully before the ThreadPool object is destroyed.
  for (std::thread &worker : _workers) 
  {
    if (worker.joinable()) 
    {
      worker.join(); // Blocks until thread completes
    }
  }
}