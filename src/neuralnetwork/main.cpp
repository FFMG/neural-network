// neuralnetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "./examples/threebitparity.h"
#include "./examples/xor.h"
#include "./libraries/instrumentor.h"

int main()
{
  MYODDWEB_PROFILE_BEGIN_SESSION( "Monitor Global", "Profile-Global.json" );

  // XOR
  ExampleXor::Xor(false);

  // Problem: 3-bit Parity
  ExampleThreebitParity::ThreebitParity();

  MYODDWEB_PROFILE_END_SESSION();

  return 0;
}
