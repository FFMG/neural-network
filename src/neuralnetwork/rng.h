#pragma once

#include <chrono>
#include <random>    // For std::mt19937 and std::random_device

class Rng
{
private:
  // Use the result_type of the engine for the seed type. This is typically unsigned int.
  using seed_type = std::mt19937::result_type;

public:

  /**
   * @brief Default constructor. Initializes with a random seed.
   *
   * Uses std::random_device to obtain a non-deterministic seed if available.
   * A time-based fallback is included for systems where random_device is not implemented.
   */
  Rng() noexcept 
    : _seed(0)
  {
    std::random_device rd;
    // Use a combination of random_device and a high-resolution clock for a more robust seed.
    // The static_cast is necessary because the types might not match.
    _seed = static_cast<seed_type>(rd()) +
      static_cast<seed_type>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    _engine.seed(_seed);
  }

  /**
   * @brief Constructor. Initializes with a specific seed.
   * @param seed The seed to use for the random number generator.
   */
  Rng(seed_type seed) noexcept : _seed(seed)
  {
    _engine.seed(_seed);
  }

  /**
   * @brief Gets the seed that was used to initialize the generator.
   * @return The seed value.
   */
  seed_type get_seed() const noexcept
  {
    return _seed;
  }

  std::mt19937 get_engine() const
  {
    return _engine;
  }
private:
  seed_type _seed;
  std::mt19937 _engine;
};