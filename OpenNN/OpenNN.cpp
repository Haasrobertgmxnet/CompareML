// opennn.cpp : 
//

#include <iostream>
#include <memory>

#include "opennn.h"

#include "MetaData.h"
#include "NeuralNetworkArchitecture.h"
#include "PathNameService.h"

using namespace opennn;
using hpc = Helper::PipelineConfig;

#ifdef _DEBUG

void layerInfo(const NeuralNetwork& neural_network) {
	// Anzahl der Layer (inklusive Eingabe- und Ausgabeschicht)
	size_t layers_number = neural_network.get_layers_number();
	std::cout << "Anzahl der Schichten: " << layers_number << std::endl;

	// Informationen über jede Schicht
	auto layers = neural_network.get_architecture();

	std::cout << "Anzahl der Layer: " << layers.size() << std::endl;
	for (size_t i = 0; i < layers.size(); ++i)
	{
		std::cout << "Layer " << i << ": " << layers[i] << " Neuronen" << std::endl;
	}
}

#endif

void opennn_pipeline(const Helper::MLCase currentCase) {
	if (!Helper::DataConfigAll[currentCase].isActive) {
		return;
	}

	// Read data
	auto pathRes = Helper::PathNameService::findFileAboveCurrentDirectory(std::string{ Helper::OpenNNDataFiles[currentCase] });
	if (!pathRes.has_value()) {
		return;
	}
	auto pathName = std::string{ pathRes.value() };
	DataSet data_set(pathName, ';', false); //';' as column separator, false means no header
	data_set.split_samples_random(Helper::PipelineConfig::train_contribution, Helper::PipelineConfig::valid_contribution, Helper::PipelineConfig::test_contribution); // Split into training, validation and test data

	const Index input_variables_number = data_set.get_input_variables_number();

	// Construct neural network architecture
	auto currentNeuralNetworkArchitecture = Helper::ConstructNeuralNetworkExample(currentCase);
	currentNeuralNetworkArchitecture.print();

	NeuralNetwork neural_network{};
	neural_network.set_inputs_number(input_variables_number);
	neural_network.set_project_type(NeuralNetwork::ProjectType::Classification);
	for (auto&& layer : currentNeuralNetworkArchitecture.Layers) {
		switch (layer.layerType) {
		case Helper::LayerType::Scaling: {
			auto current_layer = std::make_unique<ScalingLayer>(layer.inputNodes);
			neural_network.add_layer(current_layer.release());
		}
									   break;
		case Helper::LayerType::Unscaling: {
			auto current_layer = std::make_unique<UnscalingLayer>(layer.outputNodes);
			neural_network.add_layer(current_layer.release());
		}
										 break;
		case Helper::LayerType::Perceptron: {
			auto current_layer = std::make_unique<PerceptronLayer>(layer.inputNodes, layer.outputNodes);
			current_layer->set_activation_function(PerceptronLayer::ActivationFunction::RectifiedLinear);
			current_layer->set_name(layer.name);
			neural_network.add_layer(current_layer.release());
			// This releases the ownership of the uinque_ptr current_layer ...
			//... to transfer the ownership to the add_layer method of the NeuralNetwork class in OpenNN ...
			// which should care about a correct object disposal/remove.
			// !! Cave: Never do neural_network.add_layer(current_layer.get()); !!
		}
										  break;
		case Helper::LayerType::Probabilistic:
		default:
			auto current_layer = std::make_unique<ProbabilisticLayer>(layer.inputNodes, layer.outputNodes);
			current_layer->set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);
			neural_network.add_layer(current_layer.release());
		}
	}

#ifdef _DEBUG
	layerInfo(neural_network);
#endif

	// Training
	TrainingStrategy training_strategy(&neural_network, &data_set);

	training_strategy.set_maximum_epochs_number(hpc::epochs);
	training_strategy.set_display_period(hpc::epochs / 10);
	training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
	training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

	auto* adam_ptr = training_strategy.get_adaptive_moment_estimation_pointer();
	adam_ptr->set_batch_samples_number(hpc::batch_size);
	adam_ptr->set_initial_learning_rate(hpc::learning_rate);
	adam_ptr->set_maximum_epochs_number(hpc::epochs);

	try {
		auto results = TrainingResults{ training_strategy.perform_training() };
		std::cout << "Time: " << results.elapsed_time << std::endl;
		std::cout << "Epochs: " << results.get_epochs_number() << std::endl;
		std::cout << "Loss: " << results.get_loss() << std::endl;
		std::cout << "Training Error: " << results.get_training_error() << std::endl;
		std::string outpath{ "../data/opennn/" + std::string{ Helper::DataConfigAll[currentCase].name } + "_nn.xml" };
		neural_network.save(outpath);

		// Testing analysis and confusion matrix
		const TestingAnalysis testing_analysis(&neural_network, &data_set);
		const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
		std::cout << "\nConfusion matrix:\n" << confusion << std::endl;

	}
	catch (std::exception& ex) {
		std::cout << ex.what() << std::endl;
	}
}

int main() {
	// calc();
	Helper::calc(opennn_pipeline);
	return 0;
}


