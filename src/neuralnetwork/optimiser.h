#pragma once
#include <algorithm>
#include <stdexcept>
#include <string>

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
  NadamW,
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
  case OptimiserType::NadamW:    return "NadamW";
  case OptimiserType::AMSGrad:   return "AMSGrad";
  case OptimiserType::LAMB:      return "LAMB";
  case OptimiserType::Lion:      return "Lion";
  case OptimiserType::None:      return "None";
  default:
    throw std::out_of_range("Unknown OptimiserType value");
  }
}

inline OptimiserType string_to_optimiser_type(const std::string& str)
{
  std::string lower_str = str;
  // Convert the string to lowercase for case-insensitive comparison
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
    [](unsigned char c) { return std::tolower(c); });

  if (lower_str == "sgd")
  {
    return OptimiserType::SGD;
  }
  if (lower_str == "momentum")
  {
    return OptimiserType::Momentum;
  }
  if (lower_str == "nesterov")
  {
    return OptimiserType::Nesterov;
  }
  if (lower_str == "rmsprop")
  {
    return OptimiserType::RMSProp;
  }
  if (lower_str == "adam")
  {
    return OptimiserType::Adam;
  }
  if (lower_str == "adamw")
  {
    return OptimiserType::AdamW;
  }
  if (lower_str == "adagrad")
  {
    return OptimiserType::AdaGrad;
  }
  if (lower_str == "adadelta")
  {
    return OptimiserType::AdaDelta;
  }
  if (lower_str == "nadam")
  {
    return OptimiserType::Nadam;
  }
  if (lower_str == "nadamw")
  {
    return OptimiserType::NadamW;
  }
  if (lower_str == "amsgrad")
  {
    return OptimiserType::AMSGrad;
  }
  if (lower_str == "lamb")
  {
    return OptimiserType::LAMB;
  }
  if (lower_str == "lion")
  {
    return OptimiserType::Lion;
  }
  if (lower_str == "none")
  {
    return OptimiserType::None;
  }
  // If no match is found after checking all possibilities, throw an exception
  throw std::invalid_argument("Unknown optimiser type: " + str);
}