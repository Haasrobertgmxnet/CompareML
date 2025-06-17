#pragma once

// #include <string_view>
#include <vector>
#include <map>
#include <string>

#include "MetaData.h"

namespace Helper {
	std::map<MLCase, std::vector<size_t>> ArchMap = { 
		{Helper::MLCase::Iris, {4, 3, 3}}, 
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
}
