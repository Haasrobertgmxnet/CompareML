#pragma once

#include <string_view>
#include <vector>
#include <array>
#include <tuple>

namespace Helper {
	enum class MLCase : std::size_t {
		Iris = 0,
		Wine = 1,
		Ionosphere = 2,
		Cancer = 3,
		Diabetes = 4
	};
}

// Data classes strorin gth confuguration of NN structure, i.e. storing the
// * use case
// * number of nodes in the input layer
// * number of hidden layers and their number of nodes
// * number of nodes in the output layer
// * if there is a dropout layer, the dropout ratio

// * The "STATIC_AT_COMPILE_TIME" can be activated by #define STATIC_AT_COMPILE_TIME but:
// ** its usability is worse since it is built-up on a tuple structure and there would be sth left to do for improving th usability.
// ** there is no need to have all info already at compile time.
// ** So, this variant may be removed in future commits.

#define STATIC_AT_COMPILE_TIME _
// #define USE_INITIALIZER_LIST _

#ifdef STATIC_AT_COMPILE_TIME

namespace Helper {
	template<size_t N>
	struct FFNStructureData {
		constexpr FFNStructureData(const MLCase _mlCase,
			const std::string_view& _opennnFile,
			const std::string_view& _mlpackFile,
			const size_t _inputNodes,
			const std::array<size_t, N>& _hiddenNodes,
			const size_t _outputNodes,
			const double _dropoutRatio) :
			mlCase{ _mlCase },
			opennnFile{ _opennnFile },
			mlpackFile{ _mlpackFile },
			inputNodes{ _inputNodes },
			hiddenNodes{ _hiddenNodes },
			outputNodes{ _outputNodes },
			dropoutRatio{ _dropoutRatio }
		{
		}
		const MLCase mlCase{};
		const std::string_view opennnFile{};
		const std::string_view mlpackFile{};
		const size_t inputNodes{};
		const std::array<size_t, N> hiddenNodes{};
		const size_t outputNodes{};
		const double dropoutRatio{};
	};

	template<typename... Arrays>
	struct FFNStructureDataArray {
		std::tuple<Arrays...> data;
	};
}

extern constexpr Helper::FFNStructureDataArray<
	Helper::FFNStructureData<1>,
	Helper::FFNStructureData<3>,
	Helper::FFNStructureData<4>,
	Helper::FFNStructureData<4>,
	Helper::FFNStructureData<2>
> metaDataArray = {
	.data = std::tuple(
		Helper::FFNStructureData<1>{Helper::MLCase::Iris, "iris_opennn.csv", "iris_mlpack.csv",4,{3},3, 0.0},
		Helper::FFNStructureData<3>{Helper::MLCase::Wine, "wine_opennn.csv", "wine_mlpack.csv",13,{16,9,6},3, 0.0},
		Helper::FFNStructureData<4>{Helper::MLCase::Ionosphere, "ionosphere_opennn.csv", "ionosphere_mlpack.csv",34,{40, 28, 16, 8},2, 0.0},
		Helper::FFNStructureData<4>{Helper::MLCase::Cancer, "cancer_opennn.csv", "cancer_mlpack.csv",30,{36, 24, 12, 6},2, 0.0},
		Helper::FFNStructureData<2>{Helper::MLCase::Diabetes, "diabetes_opennn.csv", "diabetes_mlpack.csv",8,{12, 6},2, 0.0}

	)
};

#elif defined USE_INITIALIZER_LIST

namespace Helper {
	template<size_t N>
	struct FFNStructureData {
		constexpr FFNStructureData(const MLCase _mlCase,
			const std::string_view& _fileName,
			const size_t _inputNodes,
			const std::array<size_t, N>& _hiddenNodes,
			const size_t _outputNodes,
			const double _dropoutRatio,
			const std::initializer_list<ptrdiff_t>& d) :
			mlCase{ _mlCase },
			fileName{ _fileName },
			inputNodes{ _inputNodes },
			hiddenNodes{ _hiddenNodes },
			outputNodes{ _outputNodes },
			dropoutRatio{ _dropoutRatio },
			p{&d}
		{
		}
		const MLCase mlCase{};
		const std::string_view fileName{};
		const size_t inputNodes{};
		const std::array<size_t, N> hiddenNodes{};
		const size_t outputNodes{};
		const double dropoutRatio{};

		const std::initializer_list<ptrdiff_t>* p{};
	};

	template<typename... Arrays>
	struct FFNStructureDataArray {
		std::tuple<Arrays...> data;
	};

}

extern constexpr Helper::FFNStructureDataArray<
	Helper::FFNStructureData<2>,
	Helper::FFNStructureData<3>,
	Helper::FFNStructureData<4>,
	Helper::FFNStructureData<4>,
	Helper::FFNStructureData<2>
> metaDataArray = {
	.data = std::tuple(
		Helper::FFNStructureData<2>{Helper::MLCase::Iris, "iris.csv",4,{6,4},3, 0.0, {4, 6, 4, 3}},
		Helper::FFNStructureData<3>{Helper::MLCase::Wine, "wine.csv",13,{16,9,6},3, 0.0, {13, 16, 9, 6, 3}},
		Helper::FFNStructureData<4>{Helper::MLCase::Ionosphere, "ionosphere.csv",34,{40, 28, 16, 8},2, 0.0, {34, 40, 28, 16, 8, 2}},
		Helper::FFNStructureData<4>{Helper::MLCase::Cancer, "cancer.csv",30,{36, 24, 12, 6},2, 0.0, {30, 36, 24, 12, 6, 2}},
		Helper::FFNStructureData<2>{Helper::MLCase::Diabetes, "diabetes.csv",8,{12, 6},2, 0.0, {8, 12, 6, 2}}

	)
};

#else

namespace Helper {
	struct FFNStructureData {
		constexpr FFNStructureData(const MLCase _mlCase,
			const std::string_view& _fileName,
			const size_t _inputNodes,
			const std::vector<size_t>& _hiddenNodes,
			const size_t _outputNodes,
			const double _dropoutRatio) :
			mlCase{ _mlCase },
			fileName{ _fileName },
			inputNodes{ _inputNodes },
			hiddenNodes{ _hiddenNodes },
			outputNodes{ _outputNodes },
			dropoutRatio{ _dropoutRatio }
		{
		}
		const MLCase mlCase{};
		const std::string_view fileName{};
		const size_t inputNodes{};
		const std::vector<size_t> hiddenNodes{};
		const size_t outputNodes{};
		const double dropoutRatio{};
		const std::initializer_list<size_t>{};
	};

	struct FFNStructureDataArray {
		std::array< FFNStructureData, 5> data;
		FFNStructureData get(const MLCase _mlCase) const {
			return data[static_cast<size_t>(_mlCase)];
		}

		FFNStructureData operator [](const MLCase _mlCase) const {
			return get(_mlCase);
		}
	};
}

extern Helper::FFNStructureDataArray
metaDataArray = {
	.data = std::array<Helper::FFNStructureData, 5>({
		Helper::FFNStructureData{Helper::MLCase::Iris, "iris.csv",4,std::vector<size_t>{6,4},3, 0.0},
		Helper::FFNStructureData{Helper::MLCase::Wine, "wine.csv",13,{16,9,6},3, 0.0},
		Helper::FFNStructureData{Helper::MLCase::Ionosphere, "ionosphere.csv",34,{40, 28, 16, 8},2, 0.0},
		Helper::FFNStructureData{Helper::MLCase::Cancer, "cancer.csv",30,{36, 24, 12, 6},2, 0.0},
		Helper::FFNStructureData{Helper::MLCase::Diabetes, "diabetes.csv",8,{12, 6},2, 0.0}
		}
	)
};

#endif
