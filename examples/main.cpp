// neuralnetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "common/logger.h"

// neural network
#include "neuralnetwork.h"

// examples
#include "addingproblem.h"
#include "branched_output.h"
#include "compound_output_sandwich.h"
#include "compound_softmax.h"
#include "compound_trivial_softmax.h"
#include "copymemory.h"
#include "lstm.h"
#include "lstm_multi.h"
#include "multi_output.h"
#include "multi_output_gru.h"
#include "repro_issue.h"
#include "residualxor.h"
#include "spiral.h"
#include "syntheticsentiment.h"
#include "threebitparity.h"
#include "trivial_softmax.h"
#include "twomoon.h"
#include "xor.h"
#include "libraries/instrumentor.h"

using namespace myoddweb::nn;

int main()
{
  MYODDWEB_PROFILE_BEGIN_SESSION( "Monitor Global", "Profile-Global.json" );

  auto log_level = Logger::LogLevel::Debug;
  Logger::set_level(log_level);

  // Branched Output
  ExampleBranchedOutput::Run(log_level);

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

  // Compound Softmax (Sigmoid + 5-bucket Softmax)
  ExampleCompoundSoftmax::CompoundSoftmax(log_level);

  // Multi-Output
  ExampleMultiOutput::MultiOutput(log_level);

  // Multi-Output GRU
  // ExampleMultiOutputGru::MultiOutputGru(log_level);

  // Repro Issue (TANH x2 + SOFTMAX x5)
  ExampleReproIssue::ReproIssue(log_level);

  // Trivial softmax
  ExampleTrivialSoftmax::Run(log_level);

  // Trivial Compound softmax
  ExampleCompoundTrivialSoftmax::Run(log_level, true, true);
  
  // LSTM Test
  ExampleLstm::Run(log_level);
  // LSTM Multi Test
  ExampleLstmMulti::Run(log_level);

  MYODDWEB_PROFILE_END_SESSION();

  return 0;
}