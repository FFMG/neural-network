#pragma once

#include <chrono>
#include <cstdint>
#include <random>    // For std::mt19937 and std::random_device


namespace myoddweb::nn
{
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

  /**
   * @brief Deterministically derives an independent 64-bit sub-seed from a base
   * seed and up to two identifying indices (e.g. layer index, weight index).
   *
   * Stateless and order-independent: the result depends only on the inputs,
   * never on call order, so it is safe to call concurrently from multiple
   * threads with no synchronization. Uses the splitmix64 finalizer, which has
   * good avalanche behaviour for this kind of index-based seed derivation.
   *
   * @param base_seed The root seed (e.g. NeuralNetworkOptions::seed()).
   * @param a First identifying index (e.g. layer index, or neuron index).
   * @param b Second identifying index (e.g. weight index, or call index).
   * @return A well-distributed 64-bit derived seed.
   */
  static uint64_t derive(uint64_t base_seed, uint64_t a, uint64_t b = 0) noexcept
  {
    uint64_t x = base_seed;
    x += 0x9E3779B97F4A7C15ULL + (a * 0xBF58476D1CE4E5B9ULL);
    x += 0x94D049BB133111EBULL + (b * 0xFF51AFD7ED558CCDULL);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    return x;
  }

private:
  seed_type _seed;
  std::mt19937 _engine;
};
} // namespace myoddweb::nn
