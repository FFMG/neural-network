#pragma once

#include "../libraries/instrumentor.h"

#include "layer.h"
#include "../common/logger.h"

#include "../common/activation.h"
#include "../common/optimiser.h"
#include "outputlayerdetails.h"
#include <string>


namespace myoddweb::nn
{
class LayerDetails
{
public:
  LayerDetails(
    Layer::Architecture layer_architecture,
    unsigned layer_size,
    const activation& activation,
    double dropout,
    double weight_decay,
    OptimiserType optimiser_type,
    double momentum,
    bool use_layer_normalisation,
    unsigned attention_hidden_size,
    unsigned kernel_size,
    unsigned dilation,
    unsigned number_of_heads,
    unsigned feed_forward_hidden_size,
    unsigned vocabulary_size,
    unsigned embedding_dimension) noexcept :
    _layer_architecture(layer_architecture),
    _layer_size(layer_size),
    _activation(activation),
    _dropout(dropout),
    _weight_decay(weight_decay),
    _optimiser_type(optimiser_type),
    _momentum(momentum),
    _use_layer_normalisation(use_layer_normalisation),
    _attention_hidden_size(attention_hidden_size),
    _kernel_size(kernel_size),
    _dilation(dilation),
    _number_of_heads(number_of_heads),
    _feed_forward_hidden_size(feed_forward_hidden_size),
    _vocabulary_size(vocabulary_size),
    _embedding_dimension(embedding_dimension)
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
  }

  LayerDetails(const LayerDetails& src) noexcept :
    _layer_architecture(src._layer_architecture),
    _layer_size(src._layer_size),
    _activation(src._activation),
    _dropout(src._dropout),
    _weight_decay(src._weight_decay),
    _optimiser_type(src._optimiser_type),
    _momentum(src._momentum),
    _use_layer_normalisation(src._use_layer_normalisation),
    _attention_hidden_size(src._attention_hidden_size),
    _kernel_size(src._kernel_size),
    _dilation(src._dilation),
    _number_of_heads(src._number_of_heads),
    _feed_forward_hidden_size(src._feed_forward_hidden_size),
    _vocabulary_size(src._vocabulary_size),
    _embedding_dimension(src._embedding_dimension)
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
  }

  LayerDetails(LayerDetails&& src) noexcept :
    _layer_architecture(src._layer_architecture),
    _layer_size(src._layer_size),
    _activation(std::move(src._activation)),
    _dropout(src._dropout),
    _weight_decay(src._weight_decay),
    _optimiser_type( src._optimiser_type),
    _momentum( src._momentum),
    _use_layer_normalisation(src._use_layer_normalisation),
    _attention_hidden_size(src._attention_hidden_size),
    _kernel_size(src._kernel_size),
    _dilation(src._dilation),
    _number_of_heads(src._number_of_heads),
    _feed_forward_hidden_size(src._feed_forward_hidden_size),
    _vocabulary_size(src._vocabulary_size),
    _embedding_dimension(src._embedding_dimension)
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    src._layer_architecture = Layer::Architecture::None;
    src._layer_size = 0;
    src._dropout = 0.0;
    src._weight_decay = 0;
    src._use_layer_normalisation = false;
    src._attention_hidden_size = 0;
    src._kernel_size = 0;
    src._dilation = 0;
    src._number_of_heads = 0;
    src._feed_forward_hidden_size = 0;
    src._vocabulary_size = 0;
    src._embedding_dimension = 0;
  }

  LayerDetails& operator=(const LayerDetails& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    if (this != &src)
    {
      _layer_architecture = src._layer_architecture;
      _layer_size = src._layer_size;
      _activation = src._activation;
      _dropout = src._dropout;
      _weight_decay = src._weight_decay;
      _optimiser_type = src._optimiser_type;
      _momentum = src._momentum;
      _use_layer_normalisation = src._use_layer_normalisation;
      _attention_hidden_size = src._attention_hidden_size;
      _kernel_size = src._kernel_size;
      _dilation = src._dilation;
      _number_of_heads = src._number_of_heads;
      _feed_forward_hidden_size = src._feed_forward_hidden_size;
      _vocabulary_size = src._vocabulary_size;
      _embedding_dimension = src._embedding_dimension;
    }
    return *this;
  }

  LayerDetails& operator=(LayerDetails&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    if (this != &src)
    {
      _layer_architecture = src._layer_architecture;
      _layer_size = src._layer_size;
      _activation = std::move(src._activation);
      _dropout = src._dropout;
      _weight_decay = src._weight_decay;
      _optimiser_type = src._optimiser_type;
      _momentum = src._momentum;
      _use_layer_normalisation = src._use_layer_normalisation;
      _attention_hidden_size = src._attention_hidden_size;
      _kernel_size = src._kernel_size;
      _dilation = src._dilation;
      _number_of_heads = src._number_of_heads;
      _feed_forward_hidden_size = src._feed_forward_hidden_size;
      _vocabulary_size = src._vocabulary_size;
      _embedding_dimension = src._embedding_dimension;

      src._layer_architecture = Layer::Architecture::None;
      src._layer_size = 0;
      src._dropout = 0.0;
      src._weight_decay = 0;
      src._use_layer_normalisation = false;
      src._attention_hidden_size = 0;
      src._kernel_size = 0;
      src._dilation = 0;
      src._number_of_heads = 0;
      src._feed_forward_hidden_size = 0;
      src._vocabulary_size = 0;
      src._embedding_dimension = 0;
    }
    return *this;
  }

  virtual ~LayerDetails()
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
  }
  [[nodiscard]] inline const Layer::Architecture& get_layer_architecture() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _layer_architecture;
  }
  [[nodiscard]] inline unsigned get_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _layer_size;
  }
  [[nodiscard]] inline const activation& get_activation() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _activation;
  }
  [[nodiscard]] inline double get_dropout() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _dropout;
  }
  [[nodiscard]] inline double get_weight_decay() const  noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _weight_decay;
  }
  [[nodiscard]] inline OptimiserType get_optimiser_type() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _optimiser_type;
  }
  [[nodiscard]] inline double get_momentum() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _momentum;
  }
  [[nodiscard]] inline bool get_use_layer_normalisation() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _use_layer_normalisation;
  }
  [[nodiscard]] inline unsigned get_attention_hidden_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _attention_hidden_size;
  }
  [[nodiscard]] inline unsigned get_kernel_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _kernel_size;
  }
  [[nodiscard]] inline unsigned get_dilation() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _dilation;
  }
  [[nodiscard]] inline unsigned get_number_of_heads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _number_of_heads;
  }
  [[nodiscard]] inline unsigned get_feed_forward_hidden_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _feed_forward_hidden_size;
  }
  [[nodiscard]] inline unsigned get_vocabulary_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _vocabulary_size;
  }
  [[nodiscard]] inline unsigned get_embedding_dimension() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _embedding_dimension;
  }
private:
  Layer::Architecture _layer_architecture;
  unsigned _layer_size;
  activation _activation;
  double _dropout;
  double _weight_decay;
  OptimiserType _optimiser_type;
  double _momentum;
  bool _use_layer_normalisation;
  unsigned _attention_hidden_size;
  unsigned _kernel_size;
  unsigned _dilation;
  unsigned _number_of_heads;
  unsigned _feed_forward_hidden_size;
  unsigned _vocabulary_size;
  unsigned _embedding_dimension;
};
} // namespace myoddweb::nn
