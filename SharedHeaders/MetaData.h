#pragma once

#include <array>
#include <tuple>

namespace Helper {
	enum class MLCase : std::size_t {
		Iris = 0,
		MNIST = 1,
		Class = 2
	};
}

namespace Helper {
	template<size_t N>
	struct MetaData {
		constexpr MetaData(const std::string_view& _fileName, const size_t _inputNodes, const std::array<size_t, N>& _hiddenNodes, const size_t _outputNodes) :
			fileName{ _fileName },
			inputNodes{ _inputNodes },
			hiddenNodes{ _hiddenNodes },
			outputNodes{ _outputNodes }
		{}
		const std::string_view fileName{};
		size_t inputNodes{};
		std::array<size_t, N> hiddenNodes{};
		size_t outputNodes{};
	};

	template<typename... Arrays>
	struct MetaDataArray {
		std::tuple<Arrays...> data;
	};
}

extern const Helper::MetaDataArray<
	Helper::MetaData<2>,
	Helper::MetaData<3>,
	Helper::MetaData<1>
> metaDataArray = {
	.data = std::tuple(
		Helper::MetaData<2>{"iris.csv",4,{4,3},3},
		Helper::MetaData<3>{"iris.csv",4,{6,4,3},3},
		Helper::MetaData<1>{"iris.csv",4,{3},3}
	)
};
