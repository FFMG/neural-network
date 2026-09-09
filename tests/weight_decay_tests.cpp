#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <memory>
#include "layers/layer.h"
#include "layers/fflayer.h"
#include "layers/ffoutputlayer.h"
#include "layers/elmanrnnlayer.h"
#include "layers/lstmlayer.h"
#include "layers/grurnnlayer.h"
#include "layers/tcnlayer.h"
#include "layers/embeddinglayer.h"
#include "layers/attentionpoollayer.h"
#include "layers/selfattentionlayer.h"
#include "layers/multioutputlayer.h"
#include "layers/residualprojector.h"
#include "test_helper.h"

using namespace myoddweb::nn;

namespace
{

void verify_all_elements_near(
  const std::vector<double>& actual,
  const std::vector<double>& expected,
  double tol = 1e-9)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i)
  {
    EXPECT_NEAR(actual[i], expected[i], tol);
  }
}

void verify_all_elements_decayed(
  const std::vector<double>& after,
  const std::vector<double>& before,
  double factor,
  double tol = 1e-9)
{
  ASSERT_EQ(after.size(), before.size());
  for (size_t i = 0; i < after.size(); ++i)
  {
    EXPECT_NEAR(after[i], before[i] * factor, tol);
  }
}

} // anonymous namespace

// ============================================================================
// 1. FFLayer Weight Decay Tests
// ============================================================================

TEST(WeightDecayTests, FFLayerInitialisesDecaysCorrectly)
{
  const unsigned num_inputs = 4;
  const unsigned num_outputs = 8;
  const double weight_decay = 0.05;

  FFLayer layer(
    0,
    num_inputs,
    num_outputs,
    weight_decay,
    Layer::Role::Hidden,
    activation(activation::method::linear, 0.0),
    OptimiserType::AdamW,
    -1,
    0.0,
    nullptr,
    1,
    true,
    0.0,
    12345
  );

  const auto& w_decays = layer.get_w_decays();
  EXPECT_EQ(w_decays.size(), static_cast<size_t>(num_inputs) * num_outputs);
  for (size_t i = 0; i < w_decays.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(w_decays[i], weight_decay);
  }

  const auto& b_decays = layer.get_b_decays();
  EXPECT_EQ(b_decays.size(), static_cast<size_t>(num_outputs));
  for (size_t i = 0; i < b_decays.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(b_decays[i], 0.0);
  }
}

TEST(WeightDecayTests, FFLayerDecoupledDecayAdamW)
{
  const unsigned num_inputs = 4;
  const unsigned num_outputs = 6;
  const double weight_decay = 0.04;
  const double learning_rate = 0.02;

  FFLayer layer(
    0,
    num_inputs,
    num_outputs,
    weight_decay,
    Layer::Role::Hidden,
    activation(activation::method::linear, 0.0),
    OptimiserType::AdamW,
    -1,
    0.0,
    nullptr,
    1,
    true,
    0.0,
    42
  );

  const auto initial_weights = layer.get_w_values();
  const auto initial_biases = layer.get_b_values();

  // Apply stored gradients when gradients are zero (pure weight decay)
  layer.apply_stored_gradients(learning_rate, 1.0);

  // Expected decoupled decay: w_new = w_old * (1 - lr * weight_decay)
  const double expected_decay_factor = 1.0 - learning_rate * weight_decay;
  verify_all_elements_decayed(layer.get_w_values(), initial_weights, expected_decay_factor);

  // Biases must NEVER decay
  verify_all_elements_near(layer.get_b_values(), initial_biases);
}

TEST(WeightDecayTests, FFLayerSGDL2DecayWithoutMomentum)
{
  const unsigned num_inputs = 3;
  const unsigned num_outputs = 5;
  const double weight_decay = 0.05;
  const double learning_rate = 0.01;

  FFLayer layer(
    0,
    num_inputs,
    num_outputs,
    weight_decay,
    Layer::Role::Hidden,
    activation(activation::method::linear, 0.0),
    OptimiserType::SGD,
    -1,
    0.0,
    nullptr,
    1,
    true,
    0.0, // momentum = 0.0
    99
  );

  const auto initial_weights = layer.get_w_values();
  const auto initial_biases = layer.get_b_values();

  layer.apply_stored_gradients(learning_rate, 1.0);

  // SGD with zero gradient and zero momentum:
  // grad_eff = 0 + weight_decay * w
  // v = grad_eff
  // w = w - lr * v = w * (1 - lr * weight_decay)
  const double expected_decay_factor = 1.0 - learning_rate * weight_decay;
  verify_all_elements_decayed(layer.get_w_values(), initial_weights, expected_decay_factor);
  verify_all_elements_near(layer.get_b_values(), initial_biases);
}

// ============================================================================
// 2. FFOutputLayer Weight Decay Tests
// ============================================================================

TEST(WeightDecayTests, FFOutputLayerMultipleHeadsDifferentDecays)
{
  const unsigned num_inputs = 4;
  const double decay_head0 = 0.02;
  const double decay_head1 = 0.10;
  const double learning_rate = 0.01;

  std::vector<OutputLayerDetails> details;
  details.emplace_back(
    2,
    activation(activation::method::linear, 0.0),
    ErrorCalculation::type::mse,
    EvaluationConfig(),
    decay_head0,
    OptimiserType::AdamW,
    0.0
  );
  details.emplace_back(
    3,
    activation(activation::method::linear, 0.0),
    ErrorCalculation::type::mse,
    EvaluationConfig(),
    decay_head1,
    OptimiserType::AdamW,
    0.0
  );

  FFOutputLayer layer(
    1,
    details,
    num_inputs,
    5, // total outputs = 2 + 3
    1,
    true,
    123
  );

  const auto initial_weights = layer.get_w_values();
  const auto initial_biases = layer.get_b_values();

  layer.apply_stored_gradients(learning_rate, 1.0);

  const auto& updated_weights = layer.get_w_values();
  const unsigned num_outputs = 5;

  // Head 0 (neurons 0 and 1) should decay by (1 - lr * decay_head0)
  const double factor_head0 = 1.0 - learning_rate * decay_head0;
  for (unsigned i = 0; i < num_inputs; ++i)
  {
    for (unsigned j = 0; j < 2; ++j)
    {
      const size_t idx = i * num_outputs + j;
      EXPECT_NEAR(updated_weights[idx], initial_weights[idx] * factor_head0, 1e-9);
    }
  }

  // Head 1 (neurons 2, 3, 4) should decay by (1 - lr * decay_head1)
  const double factor_head1 = 1.0 - learning_rate * decay_head1;
  for (unsigned i = 0; i < num_inputs; ++i)
  {
    for (unsigned j = 2; j < 5; ++j)
    {
      const size_t idx = i * num_outputs + j;
      EXPECT_NEAR(updated_weights[idx], initial_weights[idx] * factor_head1, 1e-9);
    }
  }

  // Biases for both heads must not decay
  verify_all_elements_near(layer.get_b_values(), initial_biases);
}

// ============================================================================
// 3. ElmanRNNLayer Weight Decay Tests
// ============================================================================

TEST(WeightDecayTests, ElmanRNNLayerDecaysInputAndRecurrentWeights)
{
  const unsigned num_inputs = 3;
  const unsigned num_neurons = 4;
  const double weight_decay = 0.05;
  const double learning_rate = 0.02;

  ElmanRNNLayer layer(
    0,
    num_inputs,
    num_neurons,
    weight_decay,
    Layer::Role::Hidden,
    activation(activation::method::linear, 0.0),
    OptimiserType::AdamW,
    -1,
    0.0,
    nullptr,
    1,
    true,
    0.0,
    777
  );

  EXPECT_EQ(layer.get_w_decays().size(), static_cast<size_t>(num_inputs) * num_neurons);
  EXPECT_EQ(layer.get_rw_decays().size(), static_cast<size_t>(num_neurons) * num_neurons);

  const auto initial_w = layer.get_w_values();
  const auto initial_rw = layer.get_rw_values();
  const auto initial_b = layer.get_b_values();

  layer.apply_stored_gradients(learning_rate, 1.0);

  const double factor = 1.0 - learning_rate * weight_decay;
  verify_all_elements_decayed(layer.get_w_values(), initial_w, factor);
  verify_all_elements_decayed(layer.get_rw_values(), initial_rw, factor);
  verify_all_elements_near(layer.get_b_values(), initial_b);
}

// ============================================================================
// 4. LSTMLayer Weight Decay Tests
// ============================================================================

TEST(WeightDecayTests, LSTMLayerDecaysAllGateWeightsExcludingBiasesAndLayerNorm)
{
  const unsigned num_inputs = 2;
  const unsigned num_neurons = 4;
  const double weight_decay = 0.03;
  const double learning_rate = 0.01;

  LSTMLayer layer(
    0,
    num_inputs,
    num_neurons,
    weight_decay,
    Layer::Role::Hidden,
    activation(activation::method::tanh, 0.0),
    OptimiserType::AdamW,
    -1,
    0.0,
    nullptr,
    1,
    true,
    0.0,
    true, // use_layer_normalisation = true
    101
  );

  const auto init_w = layer.get_w_values();
  const auto init_rw = layer.get_rw_values();
  const auto init_fw = layer.get_f_w_values();
  const auto init_frw = layer.get_f_rw_values();
  const auto init_iw = layer.get_i_w_values();
  const auto init_irw = layer.get_i_rw_values();
  const auto init_ow = layer.get_o_w_values();
  const auto init_orw = layer.get_o_rw_values();

  const auto init_b = layer.get_b_values();
  const auto init_fb = layer.get_f_b_values();
  const auto init_ib = layer.get_i_b_values();
  const auto init_ob = layer.get_o_b_values();

  const auto init_ln_gain = layer.get_ln_c_gain_values();
  const auto init_ln_bias = layer.get_ln_c_bias_values();

  layer.apply_stored_gradients(learning_rate, 1.0);

  const double factor = 1.0 - learning_rate * weight_decay;

  // All 8 weight matrices must decay
  verify_all_elements_decayed(layer.get_w_values(), init_w, factor);
  verify_all_elements_decayed(layer.get_rw_values(), init_rw, factor);
  verify_all_elements_decayed(layer.get_f_w_values(), init_fw, factor);
  verify_all_elements_decayed(layer.get_f_rw_values(), init_frw, factor);
  verify_all_elements_decayed(layer.get_i_w_values(), init_iw, factor);
  verify_all_elements_decayed(layer.get_i_rw_values(), init_irw, factor);
  verify_all_elements_decayed(layer.get_o_w_values(), init_ow, factor);
  verify_all_elements_decayed(layer.get_o_rw_values(), init_orw, factor);

  // All 4 gate biases must NOT decay
  verify_all_elements_near(layer.get_b_values(), init_b);
  verify_all_elements_near(layer.get_f_b_values(), init_fb);
  verify_all_elements_near(layer.get_i_b_values(), init_ib);
  verify_all_elements_near(layer.get_o_b_values(), init_ob);

  // LayerNorm parameters must NOT decay
  verify_all_elements_near(layer.get_ln_c_gain_values(), init_ln_gain);
  verify_all_elements_near(layer.get_ln_c_bias_values(), init_ln_bias);
}

// ============================================================================
// 5. GRURNNLayer Weight Decay Tests
// ============================================================================

TEST(WeightDecayTests, GRURNNLayerDecaysAllGateWeightsExcludingBiasesAndLayerNorm)
{
  const unsigned num_inputs = 2;
  const unsigned num_neurons = 4;
  const double weight_decay = 0.04;
  const double learning_rate = 0.01;

  GRURNNLayer layer(
    0,
    num_inputs,
    num_neurons,
    weight_decay,
    Layer::Role::Hidden,
    activation(activation::method::tanh, 0.0),
    OptimiserType::AdamW,
    -1,
    0.0,
    nullptr,
    1,
    true,
    0.0,
    true, // use_layer_normalisation = true
    202
  );

  const auto init_w = layer.get_w_values();
  const auto init_rw = layer.get_rw_values();
  const auto init_zw = layer.get_z_w_values();
  const auto init_zrw = layer.get_z_rw_values();
  const auto init_rw_w = layer.get_r_w_values();
  const auto init_r_rw = layer.get_r_rw_values();

  const auto init_b = layer.get_b_values();
  const auto init_zb = layer.get_z_b_values();
  const auto init_rb = layer.get_r_b_values();

  const auto init_ln_gain = layer.get_ln_h_gain_values();
  const auto init_ln_bias = layer.get_ln_h_bias_values();

  layer.apply_stored_gradients(learning_rate, 1.0);

  const double factor = 1.0 - learning_rate * weight_decay;

  // All 6 weight matrices must decay
  verify_all_elements_decayed(layer.get_w_values(), init_w, factor);
  verify_all_elements_decayed(layer.get_rw_values(), init_rw, factor);
  verify_all_elements_decayed(layer.get_z_w_values(), init_zw, factor);
  verify_all_elements_decayed(layer.get_z_rw_values(), init_zrw, factor);
  verify_all_elements_decayed(layer.get_r_w_values(), init_rw_w, factor);
  verify_all_elements_decayed(layer.get_r_rw_values(), init_r_rw, factor);

  // Biases must NOT decay
  verify_all_elements_near(layer.get_b_values(), init_b);
  verify_all_elements_near(layer.get_z_b_values(), init_zb);
  verify_all_elements_near(layer.get_r_b_values(), init_rb);

  // LayerNorm parameters must NOT decay
  verify_all_elements_near(layer.get_ln_h_gain_values(), init_ln_gain);
  verify_all_elements_near(layer.get_ln_h_bias_values(), init_ln_bias);
}

// ============================================================================
// 6. TCNLayer Weight Decay Tests
// ============================================================================

TEST(WeightDecayTests, TCNLayerDecaysWeightsExcludingBiases)
{
  const unsigned num_inputs = 3;
  const unsigned layer_size = 4;
  const unsigned kernel_size = 3;
  const unsigned dilation = 1;
  const double weight_decay = 0.05;
  const double learning_rate = 0.01;

  TcnLayer layer(
    0,
    num_inputs,
    layer_size,
    kernel_size,
    dilation,
    weight_decay,
    Layer::Role::Hidden,
    activation(activation::method::linear, 0.0),
    OptimiserType::AdamW,
    -1,
    0.0,
    nullptr,
    1,
    true,
    0.0,
    303
  );

  const auto init_w = layer.get_w_values();
  const auto init_b = layer.get_b_values();

  layer.apply_stored_gradients(learning_rate, 1.0);

  const double factor = 1.0 - learning_rate * weight_decay;
  verify_all_elements_decayed(layer.get_w_values(), init_w, factor);
  verify_all_elements_near(layer.get_b_values(), init_b);
}

// ============================================================================
// 7. EmbeddingLayer Weight Decay Tests
// ============================================================================

TEST(WeightDecayTests, EmbeddingLayerDecaysEmbeddingTable)
{
  const unsigned vocab_size = 5;
  const unsigned embedding_dim = 4;
  const double weight_decay = 0.05;
  const double learning_rate = 0.02;

  EmbeddingLayer layer(
    0,
    1, // num_inputs
    vocab_size,
    embedding_dim,
    weight_decay,
    Layer::Role::Input,
    activation(activation::method::linear, 0.0),
    OptimiserType::AdamW,
    -1,
    0.0,
    nullptr,
    1,
    0.0,
    404
  );

  const auto init_w = layer.get_w_values();
  EXPECT_EQ(init_w.size(), vocab_size * embedding_dim);

  layer.apply_stored_gradients(learning_rate, 1.0);

  const double factor = 1.0 - learning_rate * weight_decay;
  verify_all_elements_decayed(layer.get_w_values(), init_w, factor);
}

// ============================================================================
// 8. AttentionPoolLayer Weight Decay Tests
// ============================================================================

TEST(WeightDecayTests, AttentionPoolLayerDecaysAttentionAndContextWeights)
{
  const unsigned input_size = 4;
  const unsigned d_a = 6;
  const double weight_decay = 0.04;
  const double learning_rate = 0.01;

  AttentionPoolLayer layer(
    0,
    input_size,
    d_a,
    weight_decay,
    Layer::Role::Hidden,
    activation(activation::method::tanh, 0.0),
    OptimiserType::AdamW,
    0.0,
    1,
    true,
    0.0,
    505
  );

  const auto init_wa = layer.get_wa_values();
  const auto init_v = layer.get_v_values();
  const auto init_ba = layer.get_ba_values();

  layer.apply_stored_gradients(learning_rate, 1.0);

  const double factor = 1.0 - learning_rate * weight_decay;
  verify_all_elements_decayed(layer.get_wa_values(), init_wa, factor);
  verify_all_elements_decayed(layer.get_v_values(), init_v, factor);

  // Attention bias must NOT decay
  verify_all_elements_near(layer.get_ba_values(), init_ba);
}

// ============================================================================
// 9. SelfAttentionLayer Weight Decay Tests
// ============================================================================

TEST(WeightDecayTests, SelfAttentionLayerDecaysProjectionsExcludingBiasesAndLayerNorm)
{
  const unsigned seq_dim = 4;
  const unsigned num_heads = 2;
  const unsigned ff_hidden = 8;
  const double weight_decay = 0.02;
  const double learning_rate = 0.01;

  SelfAttentionLayer layer(
    0,
    seq_dim,
    seq_dim,
    num_heads,
    ff_hidden,
    weight_decay,
    Layer::Role::Hidden,
    activation(activation::method::relu, 0.0),
    OptimiserType::AdamW,
    -1,
    0.0,
    nullptr,
    1,
    true,
    true, // use_layer_normalisation = true
    0.0,
    606
  );

  const auto init_wq = layer.get_wq_values();
  const auto init_wk = layer.get_wk_values();
  const auto init_wv = layer.get_wv_values();
  const auto init_wo = layer.get_wo_values();
  const auto init_ff1_w = layer.get_ff1_w_values();
  const auto init_ff2_w = layer.get_ff2_w_values();

  const auto init_bq = layer.get_bq_values();
  const auto init_bk = layer.get_bk_values();
  const auto init_bv = layer.get_bv_values();
  const auto init_bo = layer.get_bo_values();
  const auto init_ff1_b = layer.get_ff1_b_values();
  const auto init_ff2_b = layer.get_ff2_b_values();

  const auto init_ln1_gain = layer.get_ln1_gain_values();
  const auto init_ln1_bias = layer.get_ln1_bias_values();
  const auto init_ln2_gain = layer.get_ln2_gain_values();
  const auto init_ln2_bias = layer.get_ln2_bias_values();

  layer.apply_stored_gradients(learning_rate, 1.0);

  const double factor = 1.0 - learning_rate * weight_decay;

  // Projections and feed-forward weights must decay
  verify_all_elements_decayed(layer.get_wq_values(), init_wq, factor);
  verify_all_elements_decayed(layer.get_wk_values(), init_wk, factor);
  verify_all_elements_decayed(layer.get_wv_values(), init_wv, factor);
  verify_all_elements_decayed(layer.get_wo_values(), init_wo, factor);
  verify_all_elements_decayed(layer.get_ff1_w_values(), init_ff1_w, factor);
  verify_all_elements_decayed(layer.get_ff2_w_values(), init_ff2_w, factor);

  // Biases must NOT decay
  verify_all_elements_near(layer.get_bq_values(), init_bq);
  verify_all_elements_near(layer.get_bk_values(), init_bk);
  verify_all_elements_near(layer.get_bv_values(), init_bv);
  verify_all_elements_near(layer.get_bo_values(), init_bo);
  verify_all_elements_near(layer.get_ff1_b_values(), init_ff1_b);
  verify_all_elements_near(layer.get_ff2_b_values(), init_ff2_b);

  // LayerNorm parameters must NOT decay
  verify_all_elements_near(layer.get_ln1_gain_values(), init_ln1_gain);
  verify_all_elements_near(layer.get_ln1_bias_values(), init_ln1_bias);
  verify_all_elements_near(layer.get_ln2_gain_values(), init_ln2_gain);
  verify_all_elements_near(layer.get_ln2_bias_values(), init_ln2_bias);
}

// ============================================================================
// 10. MultiOutputLayer Weight Decay Tests
// ============================================================================

TEST(WeightDecayTests, MultiOutputLayerBranchWeightsDecay)
{
  const unsigned num_inputs = 4;
  const double weight_decay = 0.05;
  const double learning_rate = 0.01;

  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(
      Layer::Architecture::FF,
      3,
      activation(activation::method::linear, 0.0),
      0.0,
      weight_decay,
      OptimiserType::AdamW,
      0.0,
      false,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    )
  };

  OutputLayerDetails old(
    2,
    activation(activation::method::linear, 0.0),
    ErrorCalculation::type::mse,
    EvaluationConfig(),
    weight_decay,
    OptimiserType::AdamW,
    0.0
  );

  MultiOutputLayerDetails mold(hidden_layers, old);

  std::vector<MultiOutputLayerDetails> details = { mold };

  MultiOutputLayer layer(
    0,
    num_inputs,
    2,
    details,
    1,
    true,
    707
  );

  const auto& branch = layer.get_branches()[0];
  const auto initial_hidden_w = branch.layers[0]->get_w_values();
  const auto initial_hidden_b = branch.layers[0]->get_b_values();
  const auto initial_out_w = branch.layers[1]->get_w_values();
  const auto initial_out_b = branch.layers[1]->get_b_values();

  layer.apply_stored_gradients(learning_rate, 1.0);

  const double factor = 1.0 - learning_rate * weight_decay;

  // Weights inside branch must decay
  verify_all_elements_decayed(branch.layers[0]->get_w_values(), initial_hidden_w, factor);
  verify_all_elements_decayed(branch.layers[1]->get_w_values(), initial_out_w, factor);

  // Biases must NOT decay
  verify_all_elements_near(branch.layers[0]->get_b_values(), initial_hidden_b);
  verify_all_elements_near(branch.layers[1]->get_b_values(), initial_out_b);
}

// ============================================================================
// 11. ResidualProjector Weight Decay Tests
// ============================================================================

TEST(WeightDecayTests, ResidualProjectorAppliesWeightDecay)
{
  const unsigned in_size = 3;
  const unsigned out_size = 4;
  const double weight_decay = 0.05;
  const double learning_rate = 0.02;

  ResidualProjector projector(
    in_size,
    out_size,
    activation(activation::method::linear, 0.0),
    weight_decay,
    808
  );

  const auto initial_w = projector.get_w_values();
  const auto initial_weight_val = initial_w[0];

  // Apply zero gradient with clipping = 1.0
  projector.apply_weight_gradient(0.0, learning_rate, 0, 0, 1.0);

  // final_gradient = 0.0 + weight_decay * old_weight
  // new_weight = old_weight - lr * final_gradient = old_weight * (1 - lr * weight_decay)
  const double expected_val = initial_weight_val * (1.0 - learning_rate * weight_decay);
  EXPECT_NEAR(projector.get_w_values()[0], expected_val, 1e-9);
}
