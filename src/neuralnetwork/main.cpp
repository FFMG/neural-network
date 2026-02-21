// neuralnetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "logger.h"

// neural network
#include "neuralnetwork.h"

// examples
#include "./examples/addingproblem.h"
#include "./examples/copymemory.h"
#include "./examples/residualxor.h"
#include "./examples/spiral.h"
#include "./examples/syntheticsentiment.h"
#include "./examples/threebitparity.h"
#include "./examples/twomoon.h"
#include "./examples/xor.h"
#include "./libraries/instrumentor.h"

int main()
{
  MYODDWEB_PROFILE_BEGIN_SESSION( "Monitor Global", "Profile-Global.json" );

  auto log_level = Logger::LogLevel::Debug;
  Logger::set_level(log_level);

  // Copy Memory
  ExampleCopyMemory::MemoryCopy(log_level);

  // Spiral
  ExampleSpiral::Spiral(log_level, true);

  // Two Moon
  ExampleTwoMoon::TwoMoon(log_level, true);

  // Adding Problem
  ExampleAddingProblem::AddingProblem(log_level);

  // XOR
  ExampleXor::Xor(log_level, true);

  // Residual XOR
  ExampleResidualXor::Xor(log_level, true);

  // 3-bit Parity
  ExampleThreebitParity::ThreebitParity(log_level);

  // Synthetic Sentiment
  ExampleSyntheticSentiment::SyntheticSentiment(log_level);

  MYODDWEB_PROFILE_END_SESSION();

  return 0;
}