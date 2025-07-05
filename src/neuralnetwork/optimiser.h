#pragma once
#include <string>
#include <stdexcept>

enum class OptimiserType
{
  SGD,
  Momentum,
  Nesterov,
  RMSProp,
  Adam,
  AdamW,
  AdaGrad,
  AdaDelta,
  Nadam,
  AMSGrad,
  LAMB,
  Lion,
  None
};

inline std::string optimiser_type_to_string(OptimiserType type)
{
  switch (type) 
  {
  case OptimiserType::SGD:       return "SGD";
  case OptimiserType::Momentum:  return "Momentum";
  case OptimiserType::Nesterov:  return "Nesterov";
  case OptimiserType::RMSProp:   return "RMSProp";
  case OptimiserType::Adam:      return "Adam";
  case OptimiserType::AdamW:     return "AdamW";
  case OptimiserType::AdaGrad:   return "AdaGrad";
  case OptimiserType::AdaDelta:  return "AdaDelta";
  case OptimiserType::Nadam:     return "Nadam";
  case OptimiserType::AMSGrad:   return "AMSGrad";
  case OptimiserType::LAMB:      return "LAMB";
  case OptimiserType::Lion:      return "Lion";
  case OptimiserType::None:      return "None";
  default:
    throw std::out_of_range("Unknown OptimiserType value");
  }
}
