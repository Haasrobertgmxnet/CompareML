// #define __GENCODE_CLAUDE_AI

#ifdef __GENCODE_CLAUDE_AI

#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <armadillo>
#include <random>
#include <algorithm>
#include <iostream>


using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;

#include "MetaData.h"
using hpc = Helper::PipelineConfig;

namespace Helper {
    template<typename LossType, typename InitType>
    void inspectModel(const FFN<LossType, InitType>& model)
    {
        std::cout << "\nModellstruktur:\n";
        for (size_t i = 0; i < model.Network().size(); ++i)
        {
            auto& layer = model.Network()[i];

            std::cout << "Layer " << i << ": "
                << typeid(*layer).name() << std::endl;

            //if (layer->Parameters().n_elem > 0)
            //{
            //    std::cout << "  Parametergröße: "
            //        << layer->Parameters().n_rows << " x "
            //        << layer->Parameters().n_cols << std::endl;
            //}

            //if (layer->InputDimensions().size() > 0)
            //{
            //    std::cout << "  Input-Dimension: "
            //        << layer->InputDimensions().size() << std::endl;
            //}

            //if (layer->OutputDimensions().size() > 0)
            //{
            //    std::cout << "  Output-Dimension: "
            //        << layer->OutputDimensions().size() << std::endl;
            //}
        }
        std::cout << std::endl;
    }
}


template<typename ErrorType>
void testModel(FFN<ErrorType, GlorotInitialization> model, const mat& X, const mat& Y) {
    // Teste das Modell
    mat predictions;
    model.Predict(X, predictions);

    // Konvertiere Vorhersagen zu Klassenindizes
    urowvec predictedClasses(predictions.n_cols);
    urowvec actualClasses(Y.n_cols);

    for (size_t i = 0; i < predictions.n_cols; ++i)
    {
        predictedClasses(i) = predictions.col(i).index_max();
        actualClasses(i) = Y.col(i).index_max();
    }

    // Zeige einige Vorhersagen
    std::cout << "Einige Vorhersagen:" << std::endl;
    std::cout << "(Prediction|Correct)" << std::endl;
    for (size_t i = 0; i < std::min(size_t(50), predictedClasses.n_elem); ++i)
    {
        std::cout << "(" << predictedClasses(i) << "|" << actualClasses(i) <<"), ";
    }
    std::cout << std::endl;

    // Berechne Genauigkeit
    double accuracy = accu(predictedClasses == actualClasses) /
        static_cast<double>(actualClasses.n_elem);

    // Berechne Loss auf Testdaten
    double testLoss = 0.0;
    for (size_t i = 0; i < Y.n_cols; ++i)
    {
        vec pred = predictions.col(i);
        vec actual = Y.col(i);
        testLoss += accu(square(pred - actual));
    }
    testLoss /= Y.n_cols;

    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
    std::cout << "Mean Squared Error: " << testLoss << std::endl;
}

void randomTrainTestSplit(const mat& X, const mat& labels,
    mat& trainX, mat& trainY, mat& testX, mat& testY,
    double trainRatio = 0.8) {

    size_t totalSamples = X.n_cols;
    size_t trainSize = static_cast<size_t>(totalSamples * trainRatio);
    size_t testSize = totalSamples - trainSize;

    // Erstelle einen Vektor mit allen Indizes
    std::vector<size_t> indices(totalSamples);
    std::iota(indices.begin(), indices.end(), 0);

    // Mische die Indizes zufällig
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    // Reserviere Speicher für die Matrizen
    trainX.set_size(X.n_rows, trainSize);
    trainY.set_size(labels.n_rows, trainSize);
    testX.set_size(X.n_rows, testSize);
    testY.set_size(labels.n_rows, testSize);

    // Fülle die Trainingsmatrizen
    for (size_t i = 0; i < trainSize; ++i) {
        trainX.col(i) = X.col(indices[i]);
        trainY.col(i) = labels.col(indices[i]);
    }

    // Fülle die Testmatrizen
    for (size_t i = 0; i < testSize; ++i) {
        testX.col(i) = X.col(indices[trainSize + i]);
        testY.col(i) = labels.col(indices[trainSize + i]);
    }
}

void classifyIris()
{
    // Lade die Iris-Daten
    mat data;
    bool loaded = data::Load("../data/mlpack/iris_mlpack.csv", data, true);

    if (!loaded) {
        std::cerr << "Fehler beim Laden der Datei iris_mlpack.csv" << std::endl;
        return;
    }

    std::cout << "Daten geladen: " << data.n_rows << " Features, "
        << data.n_cols << " Samples" << std::endl;

    // Separiere Features und Labels
    mat X = data.rows(0, 3);  // Erste 4 Spalten: Features
    mat y = data.row(4);      // Letzte Spalte: Labels

    // Konvertiere Labels zu One-Hot-Encoding für 3 Klassen
    mat labels = zeros<mat>(3, y.n_cols);
    for (size_t i = 0; i < y.n_cols; ++i)
    {
        labels((int)y(0, i), i) = 1.0;
    }

    // Normalisiere die Features (Min-Max Normalisierung)
    for (size_t i = 0; i < X.n_rows; ++i)
    {
        double minVal = X.row(i).min();
        double maxVal = X.row(i).max();
        if (maxVal != minVal)
        {
            X.row(i) = (X.row(i) - minVal) / (maxVal - minVal);
        }
    }

    // Teile Daten in Training und Test auf (80/20)
    // Variablen für Train/Test-Split
    mat trainX, trainY, testX, testY;

    // Führe den zufälligen Split durch (80/20)
    randomTrainTestSplit(X, labels, trainX, trainY, testX, testY, 0.8);

    // Erstelle das neuronale Netzwerk
    FFN<MeanSquaredError, GlorotInitialization> model;
    // FFN<CrossEntropyError, GlorotInitialization> model;

    // Alternative: Explizite Dimensionsangabe für Linear-Layer
    model.Add<Linear>(trainX.n_rows);  // Input: Anzahl Features, Hidden: 8 Neuronen
    model.Add<ReLU>();                     // Aktivierungsfunktion
    model.Add<Linear>(3);
    model.Add<ReLU>();
    model.Add<Linear>(trainY.n_rows);  // Output: Anzahl Klassen
    model.Add<Softmax>();                  // 

    Helper::inspectModel(model);

    // Konfiguriere ADAM Optimizer (vereinfachte Syntax)
    //ens::Adam optimizer(0.01,     // Lernrate
    //    64,
    //    0.9,        // Beta1
    //    0.999,      // Beta2
    //    1e-8,       // Epsilon
    //    trainX.n_cols * 500,  // Max Iterationen
    //    1e-9);      // Toleranz

    // Set parameters for the Adam optimizer.
    ens::Adam optimizer(
        hpc::learning_rate,  // Step size of the optimizer.
        hpc::batch_size, // Batch size. Number of data points that are used in each
        // iteration.
        0.9,        // Exponential decay rate for the first moment estimates.
        0.999, // Exponential decay rate for the weighted infinity norm estimates.
        1e-8,  // Value used to initialise the mean squared gradient parameter.
        hpc::epochs * trainX.n_cols, // Max number of iterations.
        1e-8,           // Tolerance.
        true);

    std::cout << "Starte Training..." << std::endl;

    // Trainiere das Modell
    model.Train(trainX, trainY, optimizer);

    std::cout << "Training abgeschlossen!" << std::endl;

    // Teste das Modell

    std::cout << "Checke Trainingsdaten" << std::endl;
    testModel<>(model, trainX, trainY);

    std::cout << "Checke Testdaten" << std::endl;
    testModel<>(model, testX, testY);

}

// Hauptfunktion für Demonstration
int main()
{
    try {
        classifyIris();
    }
    catch (const std::exception& e) {
        std::cerr << "Fehler: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

#else

/**
 * Feed Forward Neural Network (FFN) for
 * five use cases:
 * 1. Iris dataset
 * 2. Wine dataset
 * 3. Ionosphere dataset
 * 4. Breast cancer dataset
 * 5. Pima diabetes dataset
 *
 * @author Robert Haas
 */
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <algorithm>
#include <mlpack.hpp>
#include "MetaData.h"
#include "NeuralNetworkArchitecture.h"
#include "PathNameService.h"

#if ((ENS_VERSION_MAJOR < 2) || \
    ((ENS_VERSION_MAJOR == 2) && (ENS_VERSION_MINOR < 13)))
#error "need ensmallen version 2.13.0 or later"
#endif

using namespace mlpack;
using hpc = Helper::PipelineConfig;

arma::Row<size_t> getLabels(arma::mat predOut)
{
    arma::Row<size_t> predLabels(predOut.n_cols);
    for (arma::uword i = 0; i < predOut.n_cols; ++i)
    {
        predLabels(i) = predOut.col(i).index_max();
    }
    return predLabels;
}

void randomTrainTestSplit(const arma::mat& X, const arma::mat& labels,
    arma::mat& trainX, arma::mat& trainY, arma::mat& testX, arma::mat& testY,
    double trainRatio = 0.8) {

    size_t totalSamples = X.n_cols;
    size_t trainSize = static_cast<size_t>(totalSamples * trainRatio);
    size_t testSize = totalSamples - trainSize;

    // Erstelle einen Vektor mit allen Indizes
    std::vector<size_t> indices(totalSamples);
    std::iota(indices.begin(), indices.end(), 0);

    // Mische die Indizes zufällig
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    // Reserviere Speicher für die Matrizen
    trainX.set_size(X.n_rows, trainSize);
    trainY.set_size(labels.n_rows, trainSize);
    testX.set_size(X.n_rows, testSize);
    testY.set_size(labels.n_rows, testSize);

    // Fülle die Trainingsmatrizen
    for (size_t i = 0; i < trainSize; ++i) {
        trainX.col(i) = X.col(indices[i]);
        trainY.col(i) = labels.col(indices[i]);
    }

    // Fülle die Testmatrizen
    for (size_t i = 0; i < testSize; ++i) {
        testX.col(i) = X.col(indices[trainSize + i]);
        testY.col(i) = labels.col(indices[trainSize + i]);
    }
}
// Standard scaler for scaling of the feature variables
// X new = ( X old - arithmetic mean ) / standard deviation
class StandardScaler {
public:
    StandardScaler(const arma::mat& _in) :
        mean{ arma::mean(_in, 1) }, std{ arma::stddev(_in, 0, 1) }
    {
        std::cout << "Constructor\n";
    }

    void transform(arma::mat& _in) {
        _in = _in.each_col() - mean;
        _in = _in.each_col() / std;
    }

    [[nodiscard]]
    arma::mat transform(const arma::mat& _in) {
        arma::mat out{_in};
        transform(out);
        return out;
    }

private:
    arma::mat mean{};
    arma::mat std{};
};

namespace Helper {
    template<typename LossType, typename InitType>
    void inspectModel(const FFN<LossType, InitType>& model)
    {
        std::cout << "\nModellstruktur:\n";
        for (size_t i = 0; i < model.Network().size(); ++i)
        {
            auto& layer = model.Network()[i];

            std::cout << "Layer " << i << ": "
                << typeid(*layer).name() << std::endl;

            //if (layer->Parameters().n_elem > 0)
            //{
            //    std::cout << "  Parametergröße: "
            //        << layer->Parameters().n_rows << " x "
            //        << layer->Parameters().n_cols << std::endl;
            //}

            //if (layer->InputDimensions().size() > 0)
            //{
            //    std::cout << "  Input-Dimension: "
            //        << layer->InputDimensions().size() << std::endl;
            //}

            //if (layer->OutputDimensions().size() > 0)
            //{
            //    std::cout << "  Output-Dimension: "
            //        << layer->OutputDimensions().size() << std::endl;
            //}
        }
        std::cout << std::endl;
    }
}

void mlpack_pipeline(const Helper::MLCase currentCase)
{
    if (!Helper::DataConfigAll[currentCase].isActive) {
        return;
    }

    // Dataset is randomly split into validation
    // and training parts in the following ratio.
    // constexpr double RATIO = 0.1;

    // Labeled dataset that contains data for training is loaded from CSV file,
    // rows represent features, columns represent data points.
    // Read data
    auto pathRes = Helper::PathNameService::findFileAboveCurrentDirectory(std::string{ Helper::MLPackDataFiles[currentCase] });
    if (!pathRes.has_value()) {
        return;
    }
    auto pathName = std::string{ pathRes.value() };

    arma::mat dataset{};
    data::Load(pathName, dataset, true);
    std::cout << "File " << std::string{ pathRes.value() } << " loaded!\n";

    // Originally on Kaggle dataset CSV file has header, so it's necessary to
    // get rid of the this row, in Armadillo representation it's the first column.
    auto has_headers = bool{ false };
    auto headerLessDataset = arma::mat{ dataset };
    if (has_headers) {
            dataset.submat(0, 1, dataset.n_rows - 1, dataset.n_cols - 1);
    }

    // Splitting the complete dataset on training and validation parts.
    arma::mat train, test;

    //data::Split(headerLessDataset, train, test, 0.2);

    //arma::mat trainX = train.submat(0, 0, train.n_rows - 2, train.n_cols - 1);
    //arma::mat testX = test.submat(0, 0, test.n_rows - 2, test.n_cols - 1);

    

    // Produce One-Hot Coding
    size_t numClasses = Helper::getOutputNodes(currentCase);

    //auto getOneHotCoding = [numClasses](const arma::mat& labels) {
    //    arma::mat trainY = arma::zeros<arma::mat>(numClasses, labels.n_elem);
    //    for (size_t i = 0; i < labels.n_elem; ++i) {
    //        trainY(labels[i], i) = 1.0;
    //    }
    //    return trainY;
    //    };

    auto X = headerLessDataset.submat(0, 0, headerLessDataset.n_rows - 2, headerLessDataset.n_cols - 1);
    auto y = headerLessDataset.submat(headerLessDataset.n_rows - 1, 0, headerLessDataset.n_rows - 1, headerLessDataset.n_cols - 1);

    // Konvertiere Labels zu One-Hot-Encoding für 3 Klassen
    arma::mat labels = zeros<arma::mat>(numClasses, y.n_cols);
    for (size_t i = 0; i < y.n_cols; ++i)
    {
        labels((int)y(0, i), i) = 1.0;
    }

    // Normalisiere die Features (Min-Max Normalisierung)
    for (size_t i = 0; i < X.n_rows; ++i)
    {
        double minVal = X.row(i).min();
        double maxVal = X.row(i).max();
        if (maxVal != minVal)
        {
            X.row(i) = (X.row(i) - minVal) / (maxVal - minVal);
        }
    }

    // Teile Daten in Training und Test auf (80/20)
    // Variablen für Train/Test-Split
    arma::mat trainX, trainY, testX, testY;
    
    // Führe den zufälligen Split durch (80/20)
    randomTrainTestSplit(X, labels, trainX, trainY, testX, testY, 0.8);

    // Scaling of the training data - necessary for numerical stability
    StandardScaler stScaler{ trainX };
    stScaler.transform(trainX);
    stScaler.transform(testX);

    // Specifying the NN model. GlorotInitialization means that
    // initial weights in neurons are a uniform gaussian distribution.
    FFN<MeanSquaredError, GlorotInitialization> model;
    
    // Input layer
    std::cout << "Input layer with " << Helper::getInputNodes(currentCase) << " nodes.\n";
    model.Add<Linear>(Helper::getInputNodes(currentCase));
    model.Add<ReLU>();

    // Add the hidden layers to the neural network
    for (auto&& item : Helper::getHiddenNodes(currentCase)) {
        std::cout << "Hidden layer with " << item << " nodes.\n";
        model.Add<Linear>(item);
        // the activation function - here it is ReLU, i.e. x for x>=0 and 0 for x<0
        model.Add<ReLU>();
    }

    // Output layer
    std::cout << "Output layer with " << Helper::getOutputNodes(currentCase) << " nodes.\n";
    model.Add<Linear>(Helper::getOutputNodes(currentCase));
    model.Add<Sigmoid>();

    Helper::inspectModel(model);
   
    std::cout << "Start training ..." << std::endl;

    // Set parameters for the Adam optimizer.
    ens::Adam optimizer(
        hpc::learning_rate,  // Step size of the optimizer.
        hpc::batch_size, // Batch size. Number of data points that are used in each
        // iteration.
        0.9,        // Exponential decay rate for the first moment estimates.
        0.999, // Exponential decay rate for the weighted infinity norm estimates.
        1e-8,  // Value used to initialise the mean squared gradient parameter.
        hpc::epochs * trainX.n_cols, // Max number of iterations.
        1e-8,           // Tolerance.
        true);

    //ens::Adam optimizer(
    //    0.005,           // learning rate
    //    32,              // batch size
    //    0.9, 0.999, 1e-8,
    //    500 * trainX.n_cols,
    //    1e-8,
    //    true);

    // Konfiguriere ADAM Optimizer (vereinfachte Syntax)
    //ens::Adam optimizer(0.001,     // Lernrate
    //    0.9,        // Beta1
    //    0.999,      // Beta2
    //    1e-8,       // Epsilon
    //    trainX.n_cols * 500,  // Max Iterationen
    //    1e-9);      // Toleranz

    std::cout << "Starte Training..." << std::endl;

    // Trainiere das Modell
    model.Train(trainX, trainY, optimizer);

    // Declare callback to store best training weights.
    //ens::StoreBestCoordinates<arma::mat> bestCoordinates;

    //// Train neural network. If this is the first iteration, weights are
    //// random, using current values as starting point otherwise.
    //std::vector<double> validationLosses{};
    //model.Train(trainX,
    //    trainY,
    //    optimizer,
    //    ens::PrintLoss(),
    //    ens::ProgressBar(),
    //    // Stop the training using Early Stop at min loss.
    //    //ens::EarlyStopAtMinLoss(
    //    //    [&](const arma::mat& /* param */)
    //    //    {
    //    //        double validationLoss = model.Evaluate(validX, validY);
    //    //        double l2Penalty = 0.0;
    //    //        for (size_t i = 0; i < model.Network().size(); ++i)
    //    //        {
    //    //            if (auto* linear = dynamic_cast<Linear*>(model.Network()[i]))
    //    //            {
    //    //                l2Penalty += arma::accu(arma::square(linear->Parameters()));
    //    //            }
    //    //        }

    //    //        double lambda = 0.0;
    //    //        validationLoss += lambda * l2Penalty;
    //    //        validationLosses.push_back(validationLoss);
    //    //        return validationLoss;
    //    //    }),
    //    // Store best coordinates (neural network weights)
    //    bestCoordinates);

    // Save the best training weights into the model.
    // model.Parameters() = bestCoordinates.BestCoordinates();

    auto getCorrectPredictions = [&model](const arma::mat& X, const arma::mat& Y) {
        arma::mat predOut;
        model.Predict(X, predOut);
        arma::Row<size_t> predLabels = arma::index_max(predOut, 0);
        arma::Row<size_t> trueLabels = arma::index_max(Y, 0);
        return arma::accu(predLabels == trueLabels);
        };

    auto getAccuracy = [&model, &getCorrectPredictions](const arma::mat& X, const arma::mat& Y) {
        return static_cast<double>(getCorrectPredictions(X, Y)) / static_cast<double>(Y.n_elem);
        };

    auto correctlyPredicted = getCorrectPredictions(trainX, trainY);
    std::cout << "Training set: Correct predictions: " << correctlyPredicted << std::endl;
    std::println("Training set: Correct predictions: {}", correctlyPredicted);
    std::println("Training set: Out of: {}", trainY.n_elem);
    double trainAccuracy = 100 * getAccuracy(trainX, trainY);
    // double validAccuracy = 100 * getAccuracy(validX, validY);
    double testAccuracy = 100 * getAccuracy(testX, testY);

    std::cout << "Accuracy: train = " << trainAccuracy << "%" << std::endl;
    std::cout << "Accuracy: test = " << testAccuracy << "%" << std::endl;

    std::cout << "Saving predicted labels to \"results.csv\" ..." << std::endl;
    // testPred.save("results.csv", arma::csv_ascii);

    std::cout << "Neural network model is saved to \"model.bin\"" << std::endl;
    std::cout << "Finished" << std::endl;
}

int main() {
    Helper::calc(mlpack_pipeline);
    return 0;
}

#endif