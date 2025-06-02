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
#include "PathNameService.h"
#include "Timer.h"

#if ((ENS_VERSION_MAJOR < 2) || \
    ((ENS_VERSION_MAJOR == 2) && (ENS_VERSION_MINOR < 13)))
#error "need ensmallen version 2.13.0 or later"
#endif

using namespace mlpack;

template<typename T>
void printModel(const T& model) {
    // to be implemented
}

arma::Row<size_t> getLabels(arma::mat predOut)
{
    arma::Row<size_t> predLabels(predOut.n_cols);
    for (arma::uword i = 0; i < predOut.n_cols; ++i)
    {
        predLabels(i) = predOut.col(i).index_max();
    }
    return predLabels;
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

int main()
{
    
    // Timer object for measuring the execution time
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
#else
    auto cas{ metaDataArray[currentCase]};
#endif

    // Dataset is randomly split into validation
    // and training parts in the following ratio.
    constexpr double RATIO = 0.1;
    // Step size of the optimizer.
    const double STEP_SIZE = 2e-4;
    // Number of data points in each iteration of SGD
    const size_t BATCH_SIZE = 64;
    // Allow up to 50 epochs, unless we are stopped early by EarlyStopAtMinLoss.
    const int EPOCHS = 400;

    // Labeled dataset that contains data for training is loaded from CSV file,
    // rows represent features, columns represent data points.
    // Pfadname-Variable für verschiedene Pfadnamen
    std::string pathName{};
    pathName = Helper::PathNameService::findFileAboveCurrentDirectory(std::string{ cas.fileName }).value();
    arma::mat dataset{};
    data::Load(pathName, dataset, true);
    std::cout << "File " << std::string{ cas.fileName } << " loaded!\n";

      // Originally on Kaggle dataset CSV file has header, so it's necessary to
      // get rid of the this row, in Armadillo representation it's the first column.
    auto has_headers = bool{ false };
    auto headerLessDataset = arma::mat{ dataset };
    if (has_headers) {
            dataset.submat(0, 1, dataset.n_rows - 1, dataset.n_cols - 1);
    }

    // Splitting the complete dataset on training and validation parts.
    arma::mat no_test, test;
    data::Split(headerLessDataset, no_test, test, 0.2);

    // Splitting the complete dataset on training and validation parts.
    arma::mat train, valid;
    data::Split(no_test, train, valid, 0.1);

    arma::mat trainX = train.submat(0, 0, train.n_rows - 2, train.n_cols - 1);
    arma::mat validX = valid.submat(0, 0, valid.n_rows - 2, valid.n_cols - 1);
    arma::mat testX = test.submat(0, 0, test.n_rows - 2, test.n_cols - 1);

    // Notwendige Skalierung der Trainingsdaten
    StandardScaler stScaler{ trainX };
    stScaler.transform(trainX);
    stScaler.transform(validX);
    stScaler.transform(testX);

    // Labels should specify the class of a data point and be in the interval [0,
    // numClasses).

    // Creating labels for training and validating dataset.
    const arma::mat trainY = train.row(train.n_rows - 1);
    const arma::mat validY = valid.row(train.n_rows - 1);
    const arma::mat testY = test.row(train.n_rows - 1);

    // Specifying the NN model. NegativeLogLikelihood is the output layer that
    // is used for classification problem. GlorotInitialization means that
    // initial weights in neurons are a uniform gaussian distribution.
    FFN<NegativeLogLikelihood, GlorotInitialization> model;
    // This is intermediate layer that is needed for connection between input
    // data and relu layer. Parameters specify the number of input features
    // and number of neurons in the next layer.

    // Add the hidden layers to the neural network
    for (auto&& item : cas.hiddenNodes) {
        std::cout << "Versteckte Schicht mit " << item << " Knoten.\n";
        model.Add<Linear>(item);
        // the activation function - here it is ReLU, i.e. x for x>=0 and 0 for x<0
        model.Add<ReLU>();
    }

    // Dropout layer for regularization. First parameter is the probability of
    // setting a specific value to 0.
    // model.Add<Dropout>(0.1);
    // Ausgabeschicht / Output layer
    std::cout << "Ausgabeschicht mit " << cas.outputNodes << " Knoten.\n";
    model.Add<Linear>(cas.outputNodes);
    // LogSoftMax layer is used together with NegativeLogLikelihood for mapping
    // output values to log of probabilities of being a specific class.
    model.Add<LogSoftMax>();

    printModel<FFN<NegativeLogLikelihood, GlorotInitialization>>(model);
    std::cout << "Start training ..." << std::endl;

    // Set parameters for the Adam optimizer.
    ens::Adam optimizer(
        STEP_SIZE,  // Step size of the optimizer.
        BATCH_SIZE, // Batch size. Number of data points that are used in each
        // iteration.
        0.9,        // Exponential decay rate for the first moment estimates.
        0.999, // Exponential decay rate for the weighted infinity norm estimates.
        1e-8,  // Value used to initialise the mean squared gradient parameter.
        EPOCHS * trainX.n_cols, // Max number of iterations.
        1e-8,           // Tolerance.
        true);

    // Declare callback to store best training weights.
    ens::StoreBestCoordinates<arma::mat> bestCoordinates;

    // Train neural network. If this is the first iteration, weights are
    // random, using current values as starting point otherwise.
    model.Train(trainX,
        trainY,
        optimizer,
        ens::PrintLoss(),
        ens::ProgressBar(),
        // Stop the training using Early Stop at min loss.
        ens::EarlyStopAtMinLoss(
            [&](const arma::mat& /* param */)
            {
                double validationLoss = model.Evaluate(validX, validY);
                double l2Penalty = 0.0;
                for (size_t i = 0; i < model.Network().size(); ++i)
                {
                    if (auto* linear = dynamic_cast<Linear*>(model.Network()[i]))
                    {
                        l2Penalty += arma::accu(arma::square(linear->Parameters()));
                    }
                }

                double lambda = 0.0;
                validationLoss += lambda * l2Penalty;
                std::cout << "Validation loss: " << validationLoss << "."
                    << std::endl;
                return validationLoss;
            }),
        // Store best coordinates (neural network weights)
        bestCoordinates);

    // Save the best training weights into the model.
    model.Parameters() = bestCoordinates.BestCoordinates();

    arma::mat predOut;
    // Getting predictions on training data points.
    model.Predict(trainX, predOut);
    // Calculating accuracy on training data points.
    arma::Row<size_t> predLabels = getLabels(predOut);
    double trainAccuracy =
        arma::accu(predLabels == trainY) / (double)trainY.n_elem * 100;
    // Getting predictions on validating data points.
    model.Predict(validX, predOut);
    // Calculating accuracy on validating data points.
    predLabels = getLabels(predOut);
    double validAccuracy =
        arma::accu(predLabels == validY) / (double)validY.n_elem * 100;

    std::cout << "Accuracy: train = " << trainAccuracy << "%,"
        << "\t valid = " << validAccuracy << "%" << std::endl;

    // data::Save("model.bin", "model", model, false);

    std::cout << "Predicting on test set..." << std::endl;
    arma::mat testPredOut;
    // Getting predictions on test data points.
    model.Predict(testX, testPredOut);
    // Generating labels for the test dataset.
    arma::Row<size_t> testPred = getLabels(testPredOut);

    double testAccuracy = arma::accu(testPred == testY) /
        static_cast<double>(testY.n_elem) * 100.0;
    std::cout << "Accuracy: test = " << testAccuracy << "%" << std::endl;

    std::cout << "Saving predicted labels to \"results.csv\" ..." << std::endl;
    testPred.save("results.csv", arma::csv_ascii);

    std::cout << "Neural network model is saved to \"model.bin\"" << std::endl;
    std::cout << "Finished" << std::endl;
}
