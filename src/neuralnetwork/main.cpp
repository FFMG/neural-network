// neuralnetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "neuralnetwork.h"

int main()
{
  auto* nn = new NeuralNetwork();
  delete nn;
  std::cout << "Hello World!\n";
}
