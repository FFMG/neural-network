#include "neuron.h"
#include <random>
#include "common/logger.h"
#include "libraries/instrumentor.h"


namespace myoddweb::nn
{
Neuron::Neuron(
  unsigned index,
  const Type& type,
  const double dropout_rate
) :
  _index(index),
  _type(type),
  _dropout_rate(dropout_rate)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
}

Neuron::Neuron(const Neuron& src)  noexcept : 
  _index(src._index),
  _type(src._type),
  _dropout_rate(src._dropout_rate)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
}

Neuron& Neuron::operator=(const Neuron& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  if (this != &src)
  {
    Clean();

    _index = src._index;
    _type = src._type;
    _dropout_rate = src._dropout_rate;
  }
  return *this;
}

Neuron::Neuron(Neuron&& src) noexcept :
  _index(src._index),
  _type(src._type),
  _dropout_rate(src._dropout_rate)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");

  src._index = 0;
  src._type = Neuron::Type::Normal;
  src._dropout_rate = 0.0;
}

Neuron& Neuron::operator=(Neuron&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  if (this != &src)
  {
    Clean();

    _index = src._index;
    _dropout_rate = src._dropout_rate;
    _type = src._type;

    src._index = 0;
    src._dropout_rate = 0.0;
    src._type = Neuron::Type::Normal;
  }
  return *this;
}

Neuron::~Neuron()
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  Clean();
}

void Neuron::Clean()
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
}



double Neuron::get_dropout_rate() const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
#if VALIDATE_DATA == 1
  if (_type != Neuron::Type::Dropout)
  {
    Logger::panic("Only dropout layers have a dropout rate.");
  }
#endif
  return _dropout_rate;
}

bool Neuron::must_randomly_drop() const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
#if VALIDATE_DATA == 1
  if (_type != Neuron::Type::Dropout)
  {
    Logger::panic("Only dropout layers choose if we must dropout.");
  }
#endif
  struct ThreadLocalRng
  {
    uint64_t state;
    ThreadLocalRng() noexcept
    {
      std::random_device rd;
      uint64_t s = rd();
      s = (s << 32) | rd();
      state = s ? s : 88172645463325252ULL;
    }
    double next_double() noexcept
    {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      return (state & 0x1FFFFFFFFFFFFFULL) * 1.1102230246251565e-16;
    }
  };
  static thread_local ThreadLocalRng rng;
  return rng.next_double() < _dropout_rate;
}



} // namespace myoddweb::nn
