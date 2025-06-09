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

		// DataSet data_set("../data/iris_plant_original.csv", ';', true);

		DataSet data_set("../data/opennn/iris_opennn.csv", ';', false);

		const Index input_variables_number = data_set.get_input_variables_number();
		const Index target_variables_number = data_set.get_target_variables_number();

		std::cout << "Input variables: " << input_variables_number << std::endl;
		std::cout << "Target variables: " << target_variables_number << std::endl;

		// Neural network

		const Index hidden_neurons_number = 3;

		NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, { input_variables_number, hidden_neurons_number, target_variables_number });

		// Training strategy

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

		neural_network.save("../data/neural_network.xml");
		neural_network.save_expression_c("../data/neural_network.c");

		layerInfo(neural_network);
		std::cout << "Bye!" << std::endl;
		return 0;
	}
	catch (const exception& e)
	{
		std::cout << e.what() << std::endl;

		return 1;
	}
}

int main()
{
	runIrisExample();
	Helper::Timer tim;

	// The use cases
	// in future versions this will be stored in a (json) config
	constexpr Helper::MLCase currentCase{ Helper::MLCase::Iris };
	//constexpr Helper::MLCase currentCase{ Helper::MLCase::Wine };
	//constexpr Helper::MLCase currentCase{ Helper::MLCase::Ionosphere };
	//constexpr Helper::MLCase currentCase{ Helper::MLCase::Cancer };
	//constexpr Helper::MLCase currentCase{ Helper::MLCase::Diabetes };

#ifdef STATIC_AT_COMPILE_TIME
	auto cas{ std::get<static_cast<size_t>(currentCase)>(metaDataArray.data) };
#elif defined USE_INITIALIZER_LIST
	auto cas{ std::get<static_cast<size_t>(currentCase)>(metaDataArray.data) };
#else
	auto cas{ metaDataArray[currentCase] };
#endif

	// auto pathName = Helper::PathNameService::findFileAboveCurrentDirectory("iris_plant_original.csv").value();
	// DataSet data_set(pathName, ';', true);

	auto pathName = std::string{};
	pathName = Helper::PathNameService::findFileAboveCurrentDirectory(std::string{ cas.opennnFile }).value();
	// DataSet data_set(pathName, ';', false);
	DataSet data_set("../data/opennn/iris_opennn.csv", ';', false);

	// data_set.split_samples_random(type(0.6), type(0.1), type(0.3));

#ifdef _DEBUG
	// printInfo(data_set);
#endif

	const Index input_variables_number = data_set.get_input_variables_number();
	const Index target_variables_number = data_set.get_target_variables_number();

	const Index hidden_neurons_number = 3;

	//{
	//	NeuralNetwork neural_network1(NeuralNetwork::ProjectType::Classification, { input_variables_number, hidden_neurons_number, target_variables_number });
	//	// neural_network1.set_project_type(NeuralNetwork::ProjectType::Classification);
	//	TrainingStrategy training_strategy(&neural_network1, &data_set);
	//	training_strategy.set_maximum_epochs_number(2000);
	//	training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
	//	training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
	//	training_strategy.perform_training();

	//	const TestingAnalysis testing_analysis(&neural_network1, &data_set);
	//	const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
	//	std::cout << "\nConfusion matrix 1:\n" << confusion << std::endl;
	//}
	
	//{
	//	NeuralNetwork neural_network2(NeuralNetwork::ProjectType::Classification, { });
	//	// neural_network2.set_project_type(NeuralNetwork::ProjectType::Classification);
	//	neural_network2.add_layer(new PerceptronLayer(input_variables_number, cas.hiddenNodes[0], PerceptronLayer::ActivationFunction::RectifiedLinear));
	//	neural_network2.add_layer(new PerceptronLayer(cas.hiddenNodes[0], cas.hiddenNodes[1], PerceptronLayer::ActivationFunction::RectifiedLinear));
	//	neural_network2.add_layer(new PerceptronLayer(cas.hiddenNodes[1], target_variables_number, PerceptronLayer::ActivationFunction::HyperbolicTangent));
	//	TrainingStrategy training_strategy(&neural_network2, &data_set);
	//	training_strategy.set_maximum_epochs_number(2000);
	//	training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
	//	training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
	//	training_strategy.perform_training();

	//	const TestingAnalysis testing_analysis(&neural_network2, &data_set);
	//	const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
	//	std::cout << "\nConfusion matrix 2:\n" << confusion << std::endl;
	//}

	NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, { });
	// NeuralNetwork neural_network{};
	neural_network.set_project_type(NeuralNetwork::ProjectType::Classification);

	neural_network.add_layer(new PerceptronLayer(input_variables_number, cas.hiddenNodes[0], PerceptronLayer::ActivationFunction::RectifiedLinear));
	for (Index j = 1; j < cas.hiddenNodes.size();++j) {
		neural_network.add_layer(new PerceptronLayer(cas.hiddenNodes[j - 1], cas.hiddenNodes[j], PerceptronLayer::ActivationFunction::RectifiedLinear));
	}
	//neural_network.add_layer(new PerceptronLayer(cas.hiddenNodes[cas.hiddenNodes.size() - 1], target_variables_number, PerceptronLayer::ActivationFunction::HyperbolicTangent));
	neural_network.add_layer(new PerceptronLayer(cas.hiddenNodes[cas.hiddenNodes.size() - 1], cas.outputNodes, PerceptronLayer::ActivationFunction::Linear));
	layerInfo(neural_network);

	TrainingStrategy training_strategy(&neural_network, &data_set);
	training_strategy.set_maximum_epochs_number(8000);
	training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
	training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
	training_strategy.perform_training();

	// Testing analysis
	const TestingAnalysis testing_analysis(&neural_network, &data_set);
	
	const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();


	Tensor<type, 2> inputs(5, neural_network.get_inputs_number());
	Tensor<type, 2> outputs(5, neural_network.get_outputs_number());

	Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);
	Tensor<Index, 1> outputs_dimensions = get_dimensions(outputs);

	inputs.setValues({ {type(5.1),type(3.5),type(1.4),type(0.2)},
					{type(7.0),type(3.2),type(4.7),type(1.4)},
					{type(6.3),type(3.3),type(6.0),type(2.5)},
					{type(6.4),type(3.2),type(4.5),type(1.5)},
					{type(6.3),type(2.7),type(4.9),type(1.8)},
		});


	outputs = neural_network.calculate_outputs(inputs.data(), inputs_dimensions);

	std::cout << "\nConfusion matrix:\n" << confusion << std::endl;

	std::cout << "\nInputs:\n" << inputs << std::endl;

	std::cout << "\nOutputs:\n" << outputs << std::endl;

	

	std::cout << "Time difference needed for program execution: " << tim.getDuration().count() << " Milliseconds.\n";
	std::cout << "END!\n";
	return 0;
}
