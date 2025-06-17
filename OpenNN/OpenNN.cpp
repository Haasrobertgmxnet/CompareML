// opennn.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
// #include <print>
#include <exception>
#include <tuple>
#include <memory>
#include "opennn.h"
#include "opennn_strings.h"
#include "MetaData.h"
#include "NeuralNetworkArchitecture.h"
#include "PathNameService.h"
#include "Timer.h"

// extern const Helper::MetaDataArray metaDataArray;

using namespace opennn;

// Fix: Explicitly convert `metaDataArray.data` to a tuple using a helper function.  

#include <tuple>  

//template <typename... MetaDataTypes>  
//std::tuple<MetaDataTypes...> convertToTuple(const Helper::FFNStructureDataArray<MetaDataTypes...>& metaDataArray) {  
//    return metaDataArray.data;  
//}  

#ifdef _DEBUG

static void printInfo(const DataSet& data_set) {

	auto& tensor = data_set.get_data();
	auto rows = tensor.dimension(0);  // Erste Dimension (Zeilen)
	auto cols = tensor.dimension(1);  // Zweite Dimension (Spalten)

	int n{ 0 };
	auto vu = data_set.get_variables_uses();
	std::vector<DataSet::VariableUse> vuv{};
	for (auto j = 0; j < cols; ++j) {
		vuv.push_back(vu(j));
		if (vu(j) != DataSet::VariableUse::Input) {
			++n;
		}
	}

	std::vector<std::vector<type>> w{};
	for (auto i = 0; i < rows; ++i) {
		std::vector<type> w0{};
		for (auto j = 0; j < cols; ++j) {
			w0.push_back(tensor(i, j));
		}
		w.push_back(w0);
	}

	std::vector<type> s{};
	for (auto j = cols - n; j < cols; ++j) {
		type s0 = 0;
		for (auto i = 0; i < rows; ++i) {
			s0 += tensor(i, j);
		}
		s.push_back(s0);
	}

	auto ts{ std::accumulate(s.begin(),s.end(),0) };
	std::cout << "Set breakpoint here:\n";
	char c{};
	std::cin >> c;
	std::cout << "Input received\n";
}

// Eigen::Tensor< Index, 1 >

#endif

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

int runIrisExample() {
	try
	{
		std::cout << "OpenNN. Iris Plant Example." << std::endl;

		srand(static_cast<unsigned>(time(nullptr)));

		// Data set

		DataSet data_set("../data/opennn/iris_opennn.csv", ';', false);

		const Index input_variables_number = data_set.get_input_variables_number();
		const Index target_variables_number = data_set.get_target_variables_number();

		std::cout << "Input variables: " << input_variables_number << std::endl;
		std::cout << "Target variables: " << target_variables_number << std::endl;

		// Neural network
		const Index hidden_neurons_number = 3;
		NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, { input_variables_number, hidden_neurons_number, target_variables_number });

		TrainingStrategy training_strategy(&neural_network, &data_set);
		training_strategy.set_maximum_epochs_number(2000);
		training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
		training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
		training_strategy.perform_training();

		// Testing analysis
		const TestingAnalysis testing_analysis(&neural_network, &data_set);

		const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

		Tensor<type, 2> inputs(3, neural_network.get_inputs_number());
		Tensor<type, 2> outputs(3, neural_network.get_outputs_number());

		Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);
		Tensor<Index, 1> outputs_dimensions = get_dimensions(outputs);

		inputs.setValues({{type(5.1),type(3.5),type(1.4),type(0.2)},
							{type(6.4),type(3.2),type(4.5),type(1.5)},
							{type(6.3),type(2.7),type(4.9),type(1.8)}});


		outputs = neural_network.calculate_outputs(inputs.data(), inputs_dimensions);
		std::cout << "\nInputs:\n" << inputs << std::endl;
		std::cout << "\nOutputs:\n" << outputs << std::endl;
		std::cout << "\nConfusion matrix:\n" << confusion << std::endl;

		// Save results
		neural_network.save("../data/opennn/iris_nn.xml");
		neural_network.save_expression_c("../data/opennn/iris_nn.c");

		std::cout << "Bye!" << std::endl;
		return 0;
	}
	catch (const exception& e)
	{
		std::cout << e.what() << std::endl;
		return 1;
	}
}

void calc() {
	for (size_t&& j : { 0,1,2,3,4 }) {
		const Helper::MLCase currentCase{ static_cast<const Helper::MLCase>(j) };
		if(!Helper::DataConfigAll[currentCase].isActive) {
			continue;
		}

		// Read data
		auto pathRes = Helper::PathNameService::findFileAboveCurrentDirectory(std::string{ Helper::OpenNNDataFiles[currentCase] });
		if (!pathRes.has_value()) {
			continue;
		}
		auto pathName = std::string{ pathRes.value() };
		DataSet data_set(pathName, ';', false);
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
		
		
		layerInfo(neural_network);
		TrainingStrategy training_strategy(&neural_network, &data_set);
		training_strategy.set_maximum_epochs_number(5000);
		training_strategy.set_display_period(100);
		training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
		training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
		try {
			auto results = TrainingResults{ training_strategy.perform_training() };
			std::cout << "Time: " << results.elapsed_time << std::endl;
			std::cout << "Epochs: " << results.get_epochs_number() << std::endl;
			std::cout << "Loss: " << results.get_loss() << std::endl;
			std::cout << "Training Error: " << results.get_training_error() << std::endl;
			std::string outpath{ "../data/opennn/" + std::string{ Helper::DataConfigAll[currentCase].name } + "_nn.xml" };
			neural_network.save(outpath);
		}
		catch (std::exception& ex) {
			std::cout << ex.what() << std::endl;
		}
	}
}

int main() {
	std::cout << "Begin\n";
	runIrisExample();
	calc();
	return 0;
}


