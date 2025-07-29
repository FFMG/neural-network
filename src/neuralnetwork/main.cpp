// neuralnetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "logger.h"

#include "./examples/residualxor.h"
#include "./examples/threebitparity.h"
#include "./examples/xor.h"
#include "./libraries/instrumentor.h"

int main()
{
  MYODDWEB_PROFILE_BEGIN_SESSION( "Monitor Global", "Profile-Global.json" );

  auto logger = Logger(Logger::LogLevel::Debug);

  // XOR
  ExampleXor::Xor(logger, true);

  // Residual XOR
  ExampleResidualXor::Xor(logger, false);

  // 3-bit Parity
  ExampleThreebitParity::ThreebitParity(logger);

  MYODDWEB_PROFILE_END_SESSION();

  return 0;
}
