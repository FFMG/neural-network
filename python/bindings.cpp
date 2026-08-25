#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "neuralnetwork.h"
#include "neuralnetworkoptions.h"
#include "common/activation.h"
#include "common/optimiser.h"
#include "common/stochasticweightaveragingdetails.h"
#include "common/lookaheaddetails.h"
#include "layers/layer.h"
#include "layers/layerdetails.h"
#include "layers/outputlayerdetails.h"
#include "layers/multioutputlayerdetails.h"
#include "helpers/errorcalculation.h"
#include "helpers/neuralnetworkserializer.h"
#include "common/logger.h"

namespace py = pybind11;
using namespace myoddweb::nn;

PYBIND11_MODULE(neuralnetwork, m) {
    m.doc() = "Python bindings for the myoddweb::nn Neural Network Library";

    // 1. Enums
    py::enum_<activation::method>(m, "ActivationMethod")
        .value("Linear", activation::method::linear)
        .value("Sigmoid", activation::method::sigmoid)
        .value("Tanh", activation::method::tanh)
        .value("Relu", activation::method::relu)
        .value("LeakyRelu", activation::method::leakyRelu)
        .value("PRelu", activation::method::PRelu)
        .value("Selu", activation::method::selu)
        .value("Swish", activation::method::swish)
        .value("Mish", activation::method::mish)
        .value("Gelu", activation::method::gelu)
        .value("QuickGelu", activation::method::quickGelu)
        .value("Elu", activation::method::elu)
        .value("Softmax", activation::method::softmax)
        .export_values();

    py::enum_<OptimiserType>(m, "OptimiserType")
        .value("SGD", OptimiserType::SGD)
        .value("Momentum", OptimiserType::Momentum)
        .value("Nesterov", OptimiserType::Nesterov)
        .value("RMSProp", OptimiserType::RMSProp)
        .value("Adam", OptimiserType::Adam)
        .value("AdamW", OptimiserType::AdamW)
        .value("AdaGrad", OptimiserType::AdaGrad)
        .value("AdaDelta", OptimiserType::AdaDelta)
        .value("Nadam", OptimiserType::Nadam)
        .value("NadamW", OptimiserType::NadamW)
        .value("AMSGrad", OptimiserType::AMSGrad)
        .value("LAMB", OptimiserType::Lamb)
        .value("Lion", OptimiserType::Lion)
        .value("RAdam", OptimiserType::RAdam)
        .value("None_", OptimiserType::None)
        .export_values();

    py::enum_<ErrorCalculation::type>(m, "ErrorCalculationType")
        .value("None_", ErrorCalculation::type::none)
        .value("HuberLoss", ErrorCalculation::type::huber_loss)
        .value("HuberDirectionLoss", ErrorCalculation::type::huber_direction_loss)
        .value("MAE", ErrorCalculation::type::mae)
        .value("MSE", ErrorCalculation::type::mse)
        .value("RMSE", ErrorCalculation::type::rmse)
        .value("NRMSE", ErrorCalculation::type::nrmse)
        .value("MAPE", ErrorCalculation::type::mape)
        .value("SMAPE", ErrorCalculation::type::smape)
        .value("WAPE", ErrorCalculation::type::wape)
        .value("DirectionalAccuracy", ErrorCalculation::type::directional_accuracy)
        .value("BCELoss", ErrorCalculation::type::bce_loss)
        .value("CrossEntropy", ErrorCalculation::type::cross_entropy)
        .value("LogCosh", ErrorCalculation::type::log_cosh)
        .value("DirectionalConfidenceScore", ErrorCalculation::type::directional_confidence_score)
        .value("PredictionCoverage", ErrorCalculation::type::prediction_coverage)
        .value("QuantileLoss", ErrorCalculation::type::quantile_loss)
        .value("PinballLoss", ErrorCalculation::type::quantile_loss)
        .export_values();

    py::enum_<Layer::Architecture>(m, "LayerArchitecture")
        .value("None_", Layer::Architecture::None)
        .value("FF", Layer::Architecture::FF)
        .value("Elman", Layer::Architecture::Elman)
        .value("Gru", Layer::Architecture::Gru)
        .value("Lstm", Layer::Architecture::Lstm)
        .value("MultiOutput", Layer::Architecture::MultiOutput)
        .value("AttentionPool", Layer::Architecture::AttentionPool)
        .value("Tcn", Layer::Architecture::Tcn)
        .value("SelfAttention", Layer::Architecture::SelfAttention)
        .value("Embedding", Layer::Architecture::Embedding)
        .export_values();

    py::enum_<Layer::Role>(m, "LayerRole")
        .value("Input", Layer::Role::Input)
        .value("Hidden", Layer::Role::Hidden)
        .value("Output", Layer::Role::Output)
        .value("MultiOutput", Layer::Role::MultiOutput)
        .export_values();

    py::enum_<Logger::LogLevel>(m, "LogLevel")
        .value("Trace", Logger::LogLevel::Trace)
        .value("Debug", Logger::LogLevel::Debug)
        .value("Info", Logger::LogLevel::Information)
        .value("Warning", Logger::LogLevel::Warning)
        .value("Error", Logger::LogLevel::Error)
        .value("Panic", Logger::LogLevel::Panic)
        .value("None_", Logger::LogLevel::None)
        .export_values();

    py::class_<Logger, std::unique_ptr<Logger, py::nodelete>>(m, "Logger")
        .def_static("set_level", &Logger::set_level)
        .def_static("get_level", &Logger::get_level)
        .def_static("trace", [](py::args args) {
            std::ostringstream oss;
            for (auto item : args) oss << py::str(item).cast<std::string>();
            Logger::trace(oss.str());
        })
        .def_static("debug", [](py::args args) {
            std::ostringstream oss;
            for (auto item : args) oss << py::str(item).cast<std::string>();
            Logger::debug(oss.str());
        })
        .def_static("info", [](py::args args) {
            std::ostringstream oss;
            for (auto item : args) oss << py::str(item).cast<std::string>();
            Logger::info(oss.str());
        })
        .def_static("warning", [](py::args args) {
            std::ostringstream oss;
            for (auto item : args) oss << py::str(item).cast<std::string>();
            Logger::warning(oss.str());
        })
        .def_static("error", [](py::args args) {
            std::ostringstream oss;
            for (auto item : args) oss << py::str(item).cast<std::string>();
            Logger::error(oss.str());
        })
        .def_static("panic", [](py::args args) {
            std::ostringstream oss;
            for (auto item : args) oss << py::str(item).cast<std::string>();
            Logger::panic(oss.str());
        });

    // 2. Structs / Helper Classes
    py::class_<activation>(m, "Activation")
        .def(py::init<const activation::method, double, double, double>())
        .def(py::init<const activation::method, double, double>(), py::arg("method"), py::arg("alpha"), py::arg("temperature") = 1.0)
        .def("activate", py::overload_cast<double>(&activation::activate, py::const_))
        .def("activate_derivative", py::overload_cast<double>(&activation::activate_derivative, py::const_))
        .def("method_to_string", py::overload_cast<>(&activation::method_to_string, py::const_))
        .def_property_readonly("method", &activation::get_method)
        .def_property_readonly("alpha", &activation::get_alpha)
        .def_property("inference_temperature", &activation::get_inference_temperature, &activation::set_inference_temperature);

    py::class_<EvaluationConfig>(m, "EvaluationConfig")
        .def(py::init<>())
        .def(py::init<double, double, double, double, bool, double, double, double, const std::vector<double>&>(),
             py::arg("neutral_tolerance"),
             py::arg("confidence_threshold"),
             py::arg("huber_delta"),
             py::arg("direction_lambda"),
             py::arg("use_direction_penalty"),
             py::arg("cross_entropy_lambda"),
             py::arg("epsilon"),
             py::arg("label_smoothing"),
             py::arg("quantiles"))
        .def_property_readonly("neutral_tolerance", &EvaluationConfig::neutral_tolerance)
        .def_property_readonly("confidence_threshold", &EvaluationConfig::confidence_threshold)
        .def_property_readonly("huber_delta", &EvaluationConfig::huber_delta)
        .def_property_readonly("direction_lambda", &EvaluationConfig::direction_lambda)
        .def_property_readonly("use_direction_penalty", &EvaluationConfig::use_direction_penalty)
        .def_property_readonly("cross_entropy_lambda", &EvaluationConfig::cross_entropy_lambda)
        .def_property_readonly("epsilon", &EvaluationConfig::epsilon)
        .def_property_readonly("label_smoothing", &EvaluationConfig::label_smoothing)
        .def_property_readonly("quantiles", &EvaluationConfig::quantiles);

    py::class_<LayerDetails>(m, "LayerDetails")
        .def(py::init<Layer::Architecture, unsigned, const activation&, double, double, OptimiserType, double, bool, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>(),
             py::arg("architecture"), py::arg("size"), py::arg("activation"), py::arg("dropout"),
             py::arg("weight_decay"), py::arg("optimiser_type"), py::arg("momentum"), py::arg("use_layer_normalisation") = false,
             py::arg("attention_hidden_size") = 0, py::arg("kernel_size") = 0, py::arg("dilation") = 0,
             py::arg("number_of_heads") = 0, py::arg("feed_forward_hidden_size") = 0,
             py::arg("vocabulary_size") = 0, py::arg("embedding_dimension") = 0)
        .def_property_readonly("architecture", &LayerDetails::get_layer_architecture)
        .def_property_readonly("size", &LayerDetails::get_size)
        .def_property_readonly("activation", &LayerDetails::get_activation)
        .def_property_readonly("dropout", &LayerDetails::get_dropout)
        .def_property_readonly("weight_decay", &LayerDetails::get_weight_decay)
        .def_property_readonly("optimiser_type", &LayerDetails::get_optimiser_type)
        .def_property_readonly("momentum", &LayerDetails::get_momentum)
        .def_property_readonly("use_layer_normalisation", &LayerDetails::get_use_layer_normalisation)
        .def_property_readonly("attention_hidden_size", &LayerDetails::get_attention_hidden_size)
        .def_property_readonly("kernel_size", &LayerDetails::get_kernel_size)
        .def_property_readonly("dilation", &LayerDetails::get_dilation)
        .def_property_readonly("number_of_heads", &LayerDetails::get_number_of_heads)
        .def_property_readonly("feed_forward_hidden_size", &LayerDetails::get_feed_forward_hidden_size)
        .def_property_readonly("vocabulary_size", &LayerDetails::get_vocabulary_size)
        .def_property_readonly("embedding_dimension", &LayerDetails::get_embedding_dimension);

    py::class_<OutputLayerDetails>(m, "OutputLayerDetails")
        .def(py::init<unsigned, const activation&, const ErrorCalculation::type&, const EvaluationConfig&, double, OptimiserType, double>())
        .def_property_readonly("size", &OutputLayerDetails::get_size)
        .def_property_readonly("activation", &OutputLayerDetails::get_activation)
        .def_property_readonly("output_error_calculation_type", &OutputLayerDetails::get_output_error_calculation_type)
        .def_property_readonly("error_evaluation_config", &OutputLayerDetails::get_error_evaluation_config)
        .def_property_readonly("weight_decay", &OutputLayerDetails::get_weight_decay)
        .def_property_readonly("optimiser_type", &OutputLayerDetails::get_optimiser_type)
        .def_property_readonly("momentum", &OutputLayerDetails::get_momentum);

    py::class_<MultiOutputLayerDetails>(m, "MultiOutputLayerDetails")
        .def(py::init<const std::vector<LayerDetails>&, const OutputLayerDetails&>())
        .def_property_readonly("hidden_layers", &MultiOutputLayerDetails::get_hidden_layers)
        .def_property_readonly("output_details", &MultiOutputLayerDetails::get_output_details);

    py::class_<StochasticWeightAveragingDetails>(m, "StochasticWeightAveragingDetails")
        .def(py::init<bool, double, double>(),
             py::arg("swa_enabled"), py::arg("swa_start_percent"), py::arg("swa_update_percent"))
        .def_property_readonly("enabled", &StochasticWeightAveragingDetails::enabled)
        .def_property_readonly("swa_enabled", &StochasticWeightAveragingDetails::swa_enabled)
        .def_property_readonly("start_percent", &StochasticWeightAveragingDetails::start_percent)
        .def_property_readonly("swa_start_percent", &StochasticWeightAveragingDetails::swa_start_percent)
        .def_property_readonly("update_percent", &StochasticWeightAveragingDetails::update_percent)
        .def_property_readonly("swa_update_percent", &StochasticWeightAveragingDetails::swa_update_percent);

    py::class_<CosineAnnealingWarmRestartsDetails>(m, "CosineAnnealingWarmRestartsDetails")
        .def(py::init<bool, int, double, double, double>(),
             py::arg("enabled"),
             py::arg("first_cycle_epochs"),
             py::arg("cycle_multiplier"),
             py::arg("minimum_learning_rate"),
             py::arg("restart_decay"))
        .def_property_readonly("enabled", &CosineAnnealingWarmRestartsDetails::enabled)
        .def_property_readonly("first_cycle_epochs", &CosineAnnealingWarmRestartsDetails::first_cycle_epochs)
        .def_property_readonly("cycle_multiplier", &CosineAnnealingWarmRestartsDetails::cycle_multiplier)
        .def_property_readonly("minimum_learning_rate", &CosineAnnealingWarmRestartsDetails::minimum_learning_rate)
        .def_property_readonly("restart_decay", &CosineAnnealingWarmRestartsDetails::restart_decay)
        .def("calculate_learning_rate", &CosineAnnealingWarmRestartsDetails::calculate_learning_rate,
             py::arg("relative_epoch"), py::arg("base_learning_rate"));

    py::class_<LookaheadDetails>(m, "LookaheadDetails")
        .def(py::init<bool, int, double>(),
             py::arg("enabled"),
             py::arg("synchronisation_period"),
             py::arg("slow_weights_step_size"))
        .def_property_readonly("enabled", &LookaheadDetails::enabled)
        .def_property_readonly("lookahead_enabled", &LookaheadDetails::lookahead_enabled)
        .def_property_readonly("synchronisation_period", &LookaheadDetails::synchronisation_period)
        .def_property_readonly("slow_weights_step_size", &LookaheadDetails::slow_weights_step_size)
        .def_property_readonly("slow_step_size", &LookaheadDetails::slow_step_size);

    py::class_<NeuralNetworkHelperMetrics>(m, "NeuralNetworkHelperMetrics")
        .def(py::init<>())
        .def(py::init<double, ErrorCalculation::type>())
        .def_property_readonly("error", &NeuralNetworkHelperMetrics::error)
        .def_property_readonly("error_type", &NeuralNetworkHelperMetrics::error_type);

    py::class_<NeuralNetworkHelper, std::shared_ptr<NeuralNetworkHelper>>(m, "NeuralNetworkHelper")
        .def_property_readonly("learning_rate", &NeuralNetworkHelper::learning_rate)
        .def_property_readonly("number_of_epoch", &NeuralNetworkHelper::number_of_epoch)
        .def_property_readonly("epoch", &NeuralNetworkHelper::epoch)
        .def_property_readonly("percent_complete", &NeuralNetworkHelper::percent_complete)
        .def_property_readonly("sample_size", &NeuralNetworkHelper::sample_size)
        .def("calculate_forecast_metric", &NeuralNetworkHelper::calculate_forecast_metric)
        .def("calculate_forecast_metrics", &NeuralNetworkHelper::calculate_forecast_metrics, py::arg("error_types"), py::arg("in_sample") = true);

    // 3. NeuralNetworkOptions
    py::class_<NeuralNetworkOptions>(m, "NeuralNetworkOptions")
        .def_static("create", py::overload_cast<const std::vector<unsigned>&>(&NeuralNetworkOptions::create))
        .def("with_has_bias", &NeuralNetworkOptions::with_has_bias)
        .def("with_output_layer_details", py::overload_cast<const std::vector<MultiOutputLayerDetails>&>(&NeuralNetworkOptions::with_output_layer_details))
        .def("with_output_layer_details", py::overload_cast<const OutputLayerDetails&>(&NeuralNetworkOptions::with_output_layer_details))
        .def("with_output_layer_details", py::overload_cast<const std::vector<OutputLayerDetails>&>(&NeuralNetworkOptions::with_output_layer_details))
        .def("with_output_layer_details", py::overload_cast<unsigned, const activation&, const ErrorCalculation::type&, OptimiserType, double>(&NeuralNetworkOptions::with_output_layer_details))
        .def("with_output_layer_details", py::overload_cast<unsigned, const activation&, const ErrorCalculation::type&, const std::vector<double>&, OptimiserType, double>(&NeuralNetworkOptions::with_output_layer_details),
             py::arg("layer_size"), py::arg("activation"), py::arg("output_error_calculation_type"), py::arg("quantiles"), py::arg("optimiser_type"), py::arg("momentum"))
        .def("with_number_of_epoch", &NeuralNetworkOptions::with_number_of_epoch)
        .def("with_batch_size", &NeuralNetworkOptions::with_batch_size)
        .def("with_data_is_unique", &NeuralNetworkOptions::with_data_is_unique)
        .def("with_progress_callback", [](NeuralNetworkOptions& self, const std::function<bool(NeuralNetworkHelper&)>& callback) {
            return self.with_progress_callback(callback);
        })
        .def("with_number_of_threads", &NeuralNetworkOptions::with_number_of_threads)
        .def("with_learning_rate", &NeuralNetworkOptions::with_learning_rate)
        .def("with_learning_rate_decay_rate", &NeuralNetworkOptions::with_learning_rate_decay_rate)
        .def("with_learning_rate_warmup", &NeuralNetworkOptions::with_learning_rate_warmup)
        .def("with_learning_rate_boost_rate", &NeuralNetworkOptions::with_learning_rate_boost_rate)
        .def("with_adaptive_learning_rates", &NeuralNetworkOptions::with_adaptive_learning_rates)
        .def("with_hidden_layers", &NeuralNetworkOptions::with_hidden_layers)
        .def("with_residual_layer_jump", &NeuralNetworkOptions::with_residual_layer_jump)
        .def("with_clip_threshold", &NeuralNetworkOptions::with_clip_threshold)
        .def("with_shuffle_training_data", &NeuralNetworkOptions::with_shuffle_training_data)
        .def("with_shuffle_bptt_batches", &NeuralNetworkOptions::with_shuffle_bptt_batches)
        .def("with_bptt_supervise_last_step_only", &NeuralNetworkOptions::with_bptt_supervise_last_step_only)
        .def("bptt_supervise_last_step_only", &NeuralNetworkOptions::bptt_supervise_last_step_only)
        .def("with_enable_bptt", &NeuralNetworkOptions::with_enable_bptt)
        .def("with_bptt_max_ticks", &NeuralNetworkOptions::with_bptt_max_ticks)
        .def("with_update_training_monitor_percent", &NeuralNetworkOptions::with_update_training_monitor_percent)
        .def("with_stochastic_weight_averaging", py::overload_cast<const StochasticWeightAveragingDetails&>(&NeuralNetworkOptions::with_stochastic_weight_averaging))
        .def("with_stochastic_weight_averaging", py::overload_cast<bool, double, double>(&NeuralNetworkOptions::with_stochastic_weight_averaging),
             py::arg("swa_enabled"), py::arg("swa_start_percent"), py::arg("swa_update_percent"))
        .def("stochastic_weight_averaging", &NeuralNetworkOptions::stochastic_weight_averaging)
        .def("with_cosine_annealing_warm_restarts", py::overload_cast<const CosineAnnealingWarmRestartsDetails&>(&NeuralNetworkOptions::with_cosine_annealing_warm_restarts))
        .def("with_cosine_annealing_warm_restarts", py::overload_cast<bool, int, double, double, double>(&NeuralNetworkOptions::with_cosine_annealing_warm_restarts),
             py::arg("enabled"), py::arg("first_cycle_epochs"), py::arg("cycle_multiplier"), py::arg("minimum_learning_rate"), py::arg("restart_decay"))
        .def("cosine_annealing_warm_restarts", &NeuralNetworkOptions::cosine_annealing_warm_restarts)
        .def("with_lookahead", py::overload_cast<const LookaheadDetails&>(&NeuralNetworkOptions::with_lookahead))
        .def("with_lookahead", py::overload_cast<bool, int, double>(&NeuralNetworkOptions::with_lookahead),
             py::arg("enabled"), py::arg("synchronisation_period"), py::arg("slow_weights_step_size"))
        .def("lookahead", &NeuralNetworkOptions::lookahead)
        .def("with_final_error_calculation_types", &NeuralNetworkOptions::with_final_error_calculation_types)
        .def("with_log_level", &NeuralNetworkOptions::with_log_level)
        .def("with_seed", &NeuralNetworkOptions::with_seed)
        .def("seed", &NeuralNetworkOptions::seed)
        .def("build", &NeuralNetworkOptions::build);

    // 4. NeuralNetwork
    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<const NeuralNetworkOptions&>())
        .def(py::init<const std::vector<unsigned>&, const activation::method&, const activation::method&>())
        .def("train", &NeuralNetwork::train, py::call_guard<py::gil_scoped_release>())
        .def("think", py::overload_cast<const std::vector<std::vector<double>>&>(&NeuralNetwork::think, py::const_))
        .def("think", py::overload_cast<const std::vector<double>&>(&NeuralNetwork::think, py::const_))
        .def("get_topology", &NeuralNetwork::get_topology)
        .def("calculate_forecast_metric", &NeuralNetwork::calculate_forecast_metric)
        .def("calculate_forecast_metrics", &NeuralNetwork::calculate_forecast_metrics, py::arg("error_types"), py::arg("in_sample") = true)
        .def("calculate_forecast_metric_all_layers", &NeuralNetwork::calculate_forecast_metric_all_layers)
        .def("calculate_forecast_metrics_all_layers", &NeuralNetwork::calculate_forecast_metrics_all_layers, py::arg("error_types"), py::arg("in_sample") = true)
        .def("get_learning_rate", &NeuralNetwork::get_learning_rate)
        .def("get_temperature", py::overload_cast<>(&NeuralNetwork::get_temperature, py::const_))
        .def("get_temperature", py::overload_cast<unsigned>(&NeuralNetwork::get_temperature, py::const_))
        .def("get_inference_temperature", &NeuralNetwork::get_inference_temperature)
        .def("get_percent_complete", &NeuralNetwork::get_percent_complete)
        .def("has_training_data", &NeuralNetwork::has_training_data)
        .def("options", py::overload_cast<>(&NeuralNetwork::options))
        .def("options", py::overload_cast<>(&NeuralNetwork::options, py::const_));

    // 5. NeuralNetworkSerializer
    py::class_<NeuralNetworkSerializer, std::unique_ptr<NeuralNetworkSerializer, py::nodelete>>(m, "NeuralNetworkSerializer")
        .def_static("load", &NeuralNetworkSerializer::load, py::return_value_policy::take_ownership)
        .def_static("save", &NeuralNetworkSerializer::save);
}
