// Licensed to Florent Guelfucci under one or more agreements.
// Florent Guelfucci licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
#pragma once
#include <chrono>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#ifdef _WIN32
  #define NOMINMAX // we don't want to use the MS version of std::min/max
                   // use #include <algorithm> rather.
  #include <windows.h>
  #define safe_getpid() GetCurrentProcessId()
  #define pid_t unsigned long
  #ifndef __PRETTY_FUNCTION__
    #define __PRETTY_FUNCTION__ __FUNCSIG__
  #endif
#else
  #include <unistd.h>
  #define safe_getpid() getpid()
#endif

/**
 * \brief how often we want to flush log data to disk.
 *        the bigger the size, the more memory will be used.
 *        it is all lost when we crash ... but then again the json doc is corrupted by then.
 */
#define MYODDWEB_PROFILE_BUFFER 1000

// go to chrome://tracing/
namespace myoddweb
{
  struct ProfileResult
  {
    std::string Name;
    std::string Category;

    std::chrono::time_point<std::chrono::steady_clock> Start;
    std::chrono::time_point<std::chrono::steady_clock> End;
    pid_t ProcessId;
    std::thread::id ThreadID;
  };

  struct InstrumentationSession
  {
    std::string Name;
  };

  class Instrumentor
  {
  public:
    Instrumentor()
      : 
      _first_event(true),
      m_CurrentSession(nullptr)
    {
    }

    void BeginSession(const std::string& name, const std::string& filepath = "results.json")
    {
      std::lock_guard lock(_mutex);
      if (m_CurrentSession)
      {
        // If there is already a current session, then close it before beginning new one.
        // Subsequent profiling output meant for the original session will end up in the
        // newly opened session instead.  That's better than having badly formatted
        // profiling output.
        //MYODDWEB_OUT("Instrumentor::BeginSession('{0}') when session '{1}' already open.", name, m_CurrentSession->Name);
        InternalEndSession();
      }
      m_OutputStream.open(filepath);
      _session_start_time = std::chrono::steady_clock::now();

      if (m_OutputStream.is_open())
      {
        m_CurrentSession = new InstrumentationSession({ name });
        WriteHeader();
      }
      else
      {
        //MYODDWEB_OUT(("Instrumentor could not open results file '{0}'.", filepath);
      }
    }

    void EndSession()
    {
      std::lock_guard lock(_mutex);
      InternalEndSession();
    }

    void WriteProfile(const ProfileResult& result)
    {
      std::stringstream json;

      json << std::setprecision(6) << std::fixed;
      if(_first_event)
      {
        json << "{";
        _first_event = false;
      }
      else
      {
        json << ",{";
      }

      auto start_since_session = std::chrono::duration_cast<std::chrono::microseconds>(result.Start - _session_start_time).count();
      auto duration_ns = std::chrono::duration_cast<std::chrono::microseconds>(result.End - result.Start).count();
      
      json << "\"cat\":\"" << (result.Category) << "\",";
      json << "\"dur\":" << duration_ns << ',';
      json << "\"name\":\"" << result.Name << "\",";
      json << "\"ph\":\"X\",";
      json << "\"pid\":" << result.ProcessId << ",";
      json << "\"tid\":" << result.ThreadID << ",";
      json << "\"ts\":" << start_since_session;
      json << "}";

      std::lock_guard lock(_mutex);
      if (m_CurrentSession)
      {
        m_OutputStream << json.str();
        m_OutputStream.flush();
      }
    }

    static Instrumentor& Get()
    {
      static Instrumentor instance;
      return instance;
    }

  private:

    void WriteHeader()
    {
      m_OutputStream << "{\"otherData\": {},";
      m_OutputStream << "\"displayTimeUnit\": \"ns\",";
      m_OutputStream << "\"traceEvents\":[";
      m_OutputStream.flush();
    }

    void WriteFooter()
    {
      m_OutputStream << "]}";
      m_OutputStream.flush();
    }

    // Note: you must already own lock on _mutex before
    // calling InternalEndSession()
    void InternalEndSession()
    {
      if (m_CurrentSession)
      {
        WriteFooter();
        m_OutputStream.close();
        delete m_CurrentSession;
        m_CurrentSession = nullptr;
      }
    }

    bool _first_event;
    std::chrono::time_point<std::chrono::steady_clock> _session_start_time;
    std::mutex _mutex;
    InstrumentationSession* m_CurrentSession;
    std::ofstream m_OutputStream;
  };

  class InstrumentationTimer final
  {
  public:
    InstrumentationTimer(const char* name, const char* category)
      : 
      m_name(name), 
      m_category(category), 
      m_stopped(false)
    {
      _thread_id = std::this_thread::get_id();
      _process_id = safe_getpid();
      _start_time = std::chrono::steady_clock::now();;
    }

    ~InstrumentationTimer()
    {
      if (!m_stopped)
        Stop();
    }

    void Stop()
    {
      auto end_time = std::chrono::steady_clock::now();

      Instrumentor::Get().WriteProfile({ m_name, m_category, _start_time, end_time, _process_id, _thread_id });
      m_stopped = true;
    }
  private:
    std::thread::id _thread_id;
    pid_t _process_id;
    const char* m_name;
    const char* m_category;
    std::chrono::time_point<std::chrono::steady_clock> _start_time;
    bool m_stopped;
  };

  namespace InstrumentorUtils {

    template <size_t N>
    struct ChangeResult
    {
      char Data[N];
    };

    template <size_t N, size_t K>
    constexpr auto CleanupOutputString(const char(&expr)[N], const char(&remove)[K])
    {
      ChangeResult<N> result = {};

      size_t srcIndex = 0;
      size_t dstIndex = 0;
      while (srcIndex < N)
      {
        size_t matchIndex = 0;
        while (matchIndex < K - 1 && srcIndex + matchIndex < N - 1 && expr[srcIndex + matchIndex] == remove[matchIndex])
          matchIndex++;
        if (matchIndex == K - 1)
          srcIndex += matchIndex;
        result.Data[dstIndex++] = expr[srcIndex] == '"' ? '\'' : expr[srcIndex];
        srcIndex++;
      }
      return result;
    }
  }
}

#ifndef MYODDWEB_PROFILE
  #ifdef NDEBUG
  /**
     * \brief turn profiling on/off, uses a _lot_ of disc space!
     *        go to chrome://tracing/
     *        open Profile-Global.json
     */
    #define MYODDWEB_PROFILE 0
  #else
    #define MYODDWEB_PROFILE 1
  #endif // DEBUG
#endif

#if MYODDWEB_PROFILE
  #ifdef NDEBUG
    #error "You cannot use profiling in release mode!"
  #endif
  #define MYODDWEB_CONCAT_IMPL(x, y) x##y
  #define MYODDWEB_CONCAT(x, y) MYODDWEB_CONCAT_IMPL(x, y)
  #define MYODDWEB_PROFILE_BEGIN_SESSION(name, filepath) ::myoddweb::Instrumentor::Get().BeginSession(name, filepath)
  #define MYODDWEB_PROFILE_END_SESSION() ::myoddweb::Instrumentor::Get().EndSession()
  #define MYODDWEB_PROFILE_SCOPE(name, category) ::myoddweb::InstrumentationTimer MYODDWEB_CONCAT(timer, __LINE__)(name, category);
  #define MYODDWEB_PROFILE_FUNCTIONF() MYODDWEB_PROFILE_SCOPE(__PRETTY_FUNCTION__, "function")
  #define MYODDWEB_PROFILE_FUNCTION(category) MYODDWEB_PROFILE_SCOPE(__PRETTY_FUNCTION__, category)
#else
  #define MYODDWEB_PROFILE_BEGIN_SESSION(name, filepath)
  #define MYODDWEB_PROFILE_END_SESSION()
  #define MYODDWEB_PROFILE_SCOPE(name, category)
  #define MYODDWEB_PROFILE_FUNCTIONF()
  #define MYODDWEB_PROFILE_FUNCTION(category)
#endif