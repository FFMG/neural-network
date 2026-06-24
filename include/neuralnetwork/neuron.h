#pragma once

namespace myoddweb::nn
{
class Neuron
{
public:
  enum class Type
  {
    Normal,
    Dropout
  };

public:
  Neuron(
    unsigned index, 
    const Type& type,
    const double dropout_rate
    );

  Neuron(const Neuron& src) noexcept;
  Neuron& operator=(const Neuron& src) noexcept;
  Neuron(Neuron&& src) noexcept;
  Neuron& operator=(Neuron&& src) noexcept;

  virtual ~Neuron();

  [[nodiscard]] inline unsigned get_index() const
  {
    return _index;
  }

  [[nodiscard]] inline const Type& get_type() const noexcept
  {
    return _type;
  }

  [[nodiscard]] inline bool is_dropout() const noexcept
  {
    return _type == Type::Dropout;
  }

  [[nodiscard]] double get_dropout_rate() const;

  [[nodiscard]] bool must_randomly_drop() const;

private:

  void Clean();

  unsigned _index;
  Type _type;
  double _dropout_rate;
};
} // namespace myoddweb::nn
