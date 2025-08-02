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

void printTargets(const DataSet& data_set) {
	Eigen::Tensor<type, 2> targets = data_set.get_target_data();

	const auto dimensions = targets.dimensions();
	const auto rows = dimensions[0];
	const auto cols = dimensions[1];

	for (size_t i = 0; i < rows; ++i)
	{

		for (size_t j = 0; j < cols; ++j)
		{
			std::cout << targets(i, j) << " ";
		}

		std::cout << endl;
	}
}

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

Eigen::VectorXd run_sample_through_network(
	NeuralNetwork& nn, const Eigen::VectorXd& input_sample)
{
	// Tensor von Input-Dimension erstellen
	Eigen::Tensor<type, 2> input_tensor(1, input_sample.size());

	for (int j = 0; j < input_sample.size(); ++j)
		input_tensor(0, j) = input_sample[j];

	// Input-Dimension beschreiben
	Eigen::Tensor<Index, 1> input_dims(2);
	input_dims(0) = 1;                  // Batch-Größe = 1
	input_dims(1) = input_sample.size();

	// Output berechnen
	auto outputs = nn.calculate_outputs(input_tensor.data(), input_dims);

	// Output (1, N) → Eigen::VectorXd
	const size_t output_dim = outputs.dimension(1);
	Eigen::VectorXd output_vector(output_dim);
	for (size_t j = 0; j < output_dim; ++j)
		output_vector[j] = outputs(0, j);

	return output_vector;
}

double calculate_accuracy(NeuralNetwork& nn, const Eigen::Tensor<type,2>& inputs, const Eigen::Tensor<type, 2>& targets) {
	const size_t samples = inputs.dimension(0);
	const size_t input_dim = inputs.dimension(1);
	const size_t output_dim = targets.dimension(1);

	size_t correct = 0;

	for (size_t i = 0; i < samples; ++i)
	{
		Eigen::VectorXd input_sample(input_dim);
		for (size_t j = 0; j < input_dim; ++j)
		{
			input_sample[j] = inputs(i, j);
		}

		const Eigen::VectorXd output = run_sample_through_network(nn, input_sample);

		// Predicted class = argmax
		int predicted_class = -1;
		double max_val = -1.0;

		for (int j = 0; j < output_dim; ++j)
		{
			if (output[j] > max_val)
			{
				max_val = output[j];
				predicted_class = j;
			}
		}

		// True class = index of 1.0 in one-hot target
		int true_class = -1;
		for (int j = 0; j < output_dim; ++j)
		{
			if (targets(i, j) == 1.0)
			{
				true_class = j;
				break;
			}
		}

		if (predicted_class == true_class)
		{
			++correct;
		}
	}

	return static_cast<double>(correct) / samples;
}

double calculate_testing_accuracy(NeuralNetwork& nn, const DataSet& data_set)
{
	const auto& inputs = data_set.get_testing_input_data();
	const auto& targets = data_set.get_testing_target_data();
	return calculate_accuracy(nn, inputs, targets);
}

double calculate_training_accuracy(NeuralNetwork& nn, const DataSet& data_set)
{
	const auto& inputs = data_set.get_training_input_data();
	const auto& targets = data_set.get_training_target_data();
	return calculate_accuracy(nn, inputs, targets);
}

void opennn_pipeline(const Helper::MLCase currentCase) {

	auto currentConfig = Helper::DataConfigAll[currentCase];
	if (!currentConfig.isActive) {
		return;
	}

	// Read data
	auto pathRes = Helper::PathNameService::findFileAboveCurrentDirectory(std::string{ Helper::OpenNNDataFiles[currentCase] });
	if (!pathRes.has_value()) {
		return;
	}
	auto pathName = std::string{ pathRes.value() };
	DataSet data_set(pathName, ';', false); //';' as column separator, false means no header
	
	// data_set.split_samples_random(Helper::PipelineConfig::train_contribution, Helper::PipelineConfig::valid_contribution, Helper::PipelineConfig::test_contribution); // Split into training, validation and test data
	data_set.split_samples_random();
	
	// Construct neural network architecture
	auto currentNeuralNetworkArchitecture = Helper::ConstructNeuralNetworkExample(currentCase);
	currentNeuralNetworkArchitecture.print();

	NeuralNetwork neural_network{};
	const Index input_variables_number = data_set.get_input_variables_number();
	neural_network.set_inputs_number(input_variables_number);
	neural_network.set_project_type(NeuralNetwork::ProjectType::Classification);
	for (auto&& layer : currentNeuralNetworkArchitecture.Layers) {
		switch (layer.layerType) {
		case Helper::LayerType::Scaling: {
			auto current_layer = std::make_unique<ScalingLayer>(layer.inputNodes);
			current_layer->set_scalers("MinimumMaximum");
			current_layer->set_min_max_range(-1.0, 1.0);
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

	// Training
	TrainingStrategy training_strategy(&neural_network, &data_set);

	training_strategy.set_maximum_epochs_number(currentConfig.epochs);
	training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
	training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

	auto* adam_ptr = training_strategy.get_adaptive_moment_estimation_pointer();
	adam_ptr->set_batch_samples_number(64);
	adam_ptr->set_initial_learning_rate(currentConfig.learningRate);
	adam_ptr->set_maximum_epochs_number(currentConfig.epochs);

#ifdef _DEBUG
	std::cout << "layerInfo(neural_network);\n";
	layerInfo(neural_network);
#endif

	try {
		Helper::Timer tim;
		auto results = TrainingResults{ training_strategy.perform_training() };
		std::cout << "Training time (ms): " << tim.getDuration() << std::endl;
		std::cout << "Time: " << results.elapsed_time << std::endl;
		std::cout << "Epochs: " << results.get_epochs_number() << std::endl;
		std::cout << "Loss: " << results.get_loss() << std::endl;
		std::cout << "Training Error: " << results.get_training_error() << std::endl;
		std::cout << "Final results and stopping condition" << std::endl;
		results.write_final_results();
		results.write_stopping_condition();
		std::string outpath{ "../data/opennn/" + std::string{ currentConfig.name } + "_nn.xml" };
		neural_network.save(outpath);

		// Testing analysis and confusion matrix
		{
			const TestingAnalysis testing_analysis(&neural_network, &data_set);
			const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
			std::cout << "\nConfusion matrix:\n" << confusion << std::endl;
			Eigen::Tensor<type, 2> targets = data_set.get_target_data();
			const auto n_labels = targets.dimensions()[1];
			auto correct_preds = size_t{ 0 };
			auto all_preds = size_t{ 0 };
			std::cout << "Testing Accuracy: " << calculate_testing_accuracy(neural_network, data_set) << std::endl;
			std::cout << "Training Accuracy: " << calculate_training_accuracy(neural_network, data_set) << std::endl;
		}
	}
	catch (std::exception& ex) {
		std::cout << ex.what() << std::endl;
	}
}

int main() {
	srand(static_cast<unsigned>(time(nullptr)));
	Helper::calc(opennn_pipeline);
	return 0;
}


