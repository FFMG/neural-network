#include "neuron.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include "logger.h"
#include <random>

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

unsigned Neuron::get_index() const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return _index;
}

double Neuron::get_dropout_rate() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  assert(_type == Neuron::Type::Dropout);
  return _dropout_rate;
}

bool Neuron::must_randomly_drop() const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  assert(_type == Neuron::Type::Dropout);
  static thread_local std::mt19937 rng(std::random_device{}());
  std::bernoulli_distribution drop(1.0 - get_dropout_rate());
  return !drop(rng);  // true means keep, false means drop
}

bool Neuron::is_dropout() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return _type == Neuron::Type::Dropout;
}

const Neuron::Type& Neuron::get_type() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return _type;
}
