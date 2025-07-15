#pragma once

#include <functional>
#include <vector>
#include <map>
#include <string>
#include <print>

#include "MetaData.h"
#include "Timer.h"

namespace Helper {
	std::map<MLCase, std::vector<size_t>> ArchMap = { 
		{MLCase::Iris, {4, 3, 3}}, 
		{Helper::MLCase::Wine, {13, 16, 9, 6, 3}},
		{Helper::MLCase::Cancer, {30, 36, 24, 12, 6, 2}},
		{Helper::MLCase::Diabetes, {8, 12, 6, 2}},
		{Helper::MLCase::Ionosphere, {34, 40, 28, 16, 8, 2}}
	};
}

namespace Helper {
	enum class LayerType : std::size_t {
		Perceptron = 0,
		Probabilistic = 1,
		Scaling = 2,
		Unscaling = 3
	};
}
namespace Helper {
	class NeuralNetworkArchitecture {
		struct NeuralNetworkLayer {
			LayerType layerType{};
			std::string name{};
			size_t inputNodes{};
			size_t outputNodes{};
		};
	public:
		NeuralNetworkArchitecture(const std::vector<size_t>& nodes) {
			NeuralNetworkLayer scal_layer{ .layerType = LayerType::Scaling, .inputNodes = *(nodes.begin()), .outputNodes = *(nodes.begin()) };
			Layers.push_back(scal_layer);
			uint8_t layer_id{ 0 };
			for (auto it = nodes.begin(); it + 2 != nodes.end(); ++it) {
				std::string name = "Perceptron_" + std::to_string(layer_id);
				auto w = *it;
				NeuralNetworkLayer layer{ .layerType = LayerType::Perceptron, .name = name, .inputNodes = *it, .outputNodes = *(it + 1)};
				Layers.push_back(layer);
			}
			auto it = nodes.end();
			std::advance(it, -2);
			NeuralNetworkLayer layer{ .layerType = LayerType::Probabilistic, .name = "Probabilistic", .inputNodes = *it, .outputNodes = *(it + 1)};
			Layers.push_back(layer);
		}
		void print() {
			std::cout << "My neural network\n";
		}
		std::vector<NeuralNetworkLayer> Layers{};
	};
}

namespace Helper {
	NeuralNetworkArchitecture ConstructNeuralNetworkExample(const MLCase mlCase) {
		NeuralNetworkArchitecture neuralNetwork(ArchMap[mlCase]);
		return neuralNetwork;
	}

	size_t getInputNodes(const MLCase mlCase) {
		return ArchMap[mlCase].front();
	}

	size_t getOutputNodes(const MLCase mlCase) {
		return ArchMap[mlCase].back();
	}

	std::vector<size_t> getHiddenNodes(const MLCase mlCase) {
		if (ArchMap[mlCase].size() > 2) {
			return std::vector<size_t>(ArchMap[mlCase].begin() + 1, ArchMap[mlCase].end() - 1);
		}
		return std::vector<size_t>{};
	}

	void calc(const std::function<void(const Helper::MLCase currentCase)>& pipeline) {
		// Timer object for measuring the execution time
		Timer tim;
		std::vector<uint8_t> treated_cases{};
		for (uint8_t&& j : { 0, 1, 3, 4, 5 }) {
			const MLCase currentCase{ static_cast<const Helper::MLCase>(j) };
			if (!DataConfigAll[currentCase].isActive) {
				continue;
			}
			pipeline(currentCase);
			treated_cases.push_back(j);
		}
		for (auto&& item : treated_cases) {
			std::print("Case: {} ", item);
		}
		std::print("\nPipelines finished. ");
	}
}
