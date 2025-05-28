#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <algorithm>
// #include <Eigen/Dense>

namespace fs = std::filesystem;

namespace Helper {
    class PathNameService {
    public:
        static std::optional<std::string>
            findFileInCurrentDirectory(const std::string& filename)
        {
            fs::path currentDir = fs::current_path() / "..";

            for (const auto& entry : fs::directory_iterator(currentDir))
            {
                if (entry.is_regular_file() && entry.path().filename() == filename)
                {
                    // Rückgabe: relativer Pfad zum Arbeitsverzeichnis
                    return entry.path().string();
                    // Oder absolut:
                    // return fs::absolute(entry.path()).string();
                }
            }

            // Datei nicht gefunden
            return std::nullopt;
        }

        static std::optional<std::string>
            findFileAboveCurrentDirectory(const std::string& filename)
        {
            // Starte im übergeordneten Verzeichnis
            fs::path startDir = fs::current_path().parent_path();

            // Durchsuche rekursiv alle Unterverzeichnisse
            for (const auto& entry : fs::recursive_directory_iterator(startDir))
            {
                auto tmp{ entry.path().filename().string() };
                if (entry.path().filename().string() == filename)
                {
                    return entry.path()
                        .string(); // Oder fs::absolute(entry.path()).string()
                }
            }

            return std::nullopt; // Datei nicht gefunden
        }
    };
}

namespace Helper {

    template<typename T>
    Eigen::Tensor<T, 2> readCSVToTensor2D(const std::string& filename, const char& sep = ',') {
        std::ifstream file(filename);
        std::string line;
        std::vector<std::vector<T>> data;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string val;
            std::vector<T> row;
            while (std::getline(ss, val, sep)) {
                row.push_back(static_cast<T>(std::stod(val))); // Ensure type conversion
            }
            data.push_back(row);
        }

        // Bestimme Größe
        const size_t rows = data.size();
        if (rows == 0) {
            throw std::runtime_error("CSV file is empty or improperly formatted.");
        }
        const size_t cols = data[0].size();

        Eigen::Tensor<T, 2> tensor(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            if (data[i].size() != cols) {
                throw std::runtime_error("Inconsistent row sizes in CSV file.");
            }
            for (size_t j = 0; j < cols; ++j) {
                tensor(i, j) = data[i][j];
            }
        }

        return tensor;
    }

    template<typename T>
    Eigen::Tensor<T, 2> createOneHotCoding(const Eigen::Tensor<T, 2>& data) {
        auto rows{ data.dimension(0) };
        auto cols{ data.dimension(1) };

        auto w = std::vector<T>{ };
        for (size_t i = 0; i < rows; ++i) {
            auto tmp{ data(i,0) };
            if (std::find(w.begin(), w.end(), tmp) != w.end()) {
                continue;
            }
            w.push_back(tmp);
        }

        Eigen::Tensor<T, 2> tensor(rows, cols + w.size());
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                tensor(i, j) = data(i, j);
            }
        }

        // do the One-Hot-Coding
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < w.size(); ++j) {
                tensor(i, cols + j) = 0.f;
            }
            auto j = std::distance(w.begin(), std::find(w.begin(), w.end(), data(i, 0)));
            tensor(i, cols + j) = 1.f;
        }

        return tensor;
    }

}

