// opennn.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
// #include <print>
#include <exception>
#include <tuple>
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

static void look(const DataSet& data_set) {

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

#endif

int main()
{
	Helper::Timer tim;
	auto cas{ std::get<static_cast<size_t>(Helper::MLCase::Iris)>(metaDataArray.data) };

	auto pathName = std::string{};
	pathName = Helper::PathNameService::findFileAboveCurrentDirectory("iris_plant_original.csv").value();
	DataSet data_set(pathName, ';', true);

#ifdef _DEBUG
	look(data_set);
#endif

	const Index input_variables_number = data_set.get_input_variables_number();
	const Index target_variables_number = data_set.get_target_variables_number();

	const Index hidden_neurons_number = 3;

	NeuralNetwork neural_network(NeuralNetwork::ProjectType::Approximation, { input_variables_number, hidden_neurons_number, target_variables_number });

	neural_network.print();

	TrainingStrategy training_strategy(&neural_network, &data_set);
	training_strategy.set_maximum_epochs_number(5);
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

	inputs.setValues({ {type(5.1),type(3.5),type(1.4),type(0.2)},
						{type(6.4),type(3.2),type(4.5),type(1.5)},
						{type(6.3),type(2.7),type(4.9),type(1.8)} });


	outputs = neural_network.calculate_outputs(inputs.data(), inputs_dimensions);

	cout << "\nInputs:\n" << inputs << endl;

	cout << "\nOutputs:\n" << outputs << endl;

	cout << "\nConfusion matrix:\n" << confusion << endl;

	std::cout << "Time difference needed for program execution: " << tim.getDuration().count() << " Milliseconds.\n";
	std::cout << "END!\n";
	return 0;
}
