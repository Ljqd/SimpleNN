#include "Sequential.h"

#include "../layers/Dropout.h"

#include <iostream>

// Add a layer to the model
void Sequential::addLayer(std::shared_ptr<AbstractLayer> layer)
{
    layers.push_back(layer);
}

void Sequential::compile(std::shared_ptr<AbstractCost> loss)
{
    lossFunction = loss;

    // Optionally, set other attributes like optimizer and metrics here.
}

Eigen::MatrixXf Sequential::forward(const Eigen::MatrixXf& input)
{
    Eigen::MatrixXf currentOutput = input;
    for (auto& layer : layers)
    {
        currentOutput = layer->forward(currentOutput);
    }
    return currentOutput;
}

void Sequential::backward(const Eigen::MatrixXf& dOutput, double learningRate)
{
    Eigen::MatrixXf currentGradient = dOutput;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it)
    {
        currentGradient = (*it)->backward(currentGradient);
    }
}

void Sequential::train(const Eigen::MatrixXf& data,
    const Eigen::MatrixXf& target,
    int epochs,
    double learningRate,
    int batchSize)
{
    int nSamples = data.rows();
    int nBatches;

    std::pair<int, int> batchSizeAndNumber = getBatchSize(batchSize, nSamples);

    batchSize = batchSizeAndNumber.first;
    nBatches = batchSizeAndNumber.second;


    // ToDo: we need to add some sort of check
    // call this function only if we actually have Dropout in layers
    // Add flag in addLayer().
    setDropoutOnTrain();

    // for random shuffle
    std::random_device rd;
    std::mt19937 g(rd());

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double averageLoss = 0.0;

        // Shuffling mechanism
        std::vector<int> indices(nSamples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g);

        for (int batch = 0; batch < nBatches; ++batch)
        {
            int startIdx = batch * batchSize;
            int endIdx = std::min(startIdx + batchSize, nSamples); // ensure we don't go out of bounds
            int currentBatchSize = endIdx - startIdx;

            Eigen::MatrixXf batchData = data.middleRows(startIdx, currentBatchSize);
            Eigen::MatrixXf batchTarget = target.middleRows(startIdx, currentBatchSize);

            // Forward pass
            Eigen::MatrixXf predictions = forward(batchData);
            averageLoss += lossFunction->forward(predictions, batchTarget) * currentBatchSize;  // weighted by the current batch size

            Eigen::MatrixXf dOutput = lossFunction->backward(predictions, batchTarget);

            // Backward pass
            backward(dOutput, learningRate);

            // Update weights
            for (auto& layer : layers)
            {
                layer->updateWeights(learningRate);
            }
        }

        averageLoss /= nSamples;  // get the average loss
        std::cout << "Epoch " << epoch + 1 << ": Loss = " << averageLoss << std::endl;
    }
}

Eigen::MatrixXf Sequential::predict(const Eigen::MatrixXf& input)
{
    setDropoutOnPredict();
    return forward(input);
}

void Sequential::saveModel(const std::string& filepath)
{
    // Implementation to save the model's parameters to a file.
    // This could involve serializing the weight and bias matrices for each layer, among other things.
}

void Sequential::loadModel(const std::string& filepath)
{
    // Implementation to load the model's parameters from a file.
    // This would involve deserializing the saved data into weight and bias matrices and other parameters.
}

std::pair<int, int> Sequential::getBatchSize(int batchSize, int nSamples)
{
    int nBatches;
    if (batchSize > 0)
    {
        nBatches = (nSamples + batchSize - 1) / batchSize;
    }
    else
    {
        nBatches = 1;
        batchSize = nSamples;
    }
    std::pair<int, int> batchSizeAndNumber = { batchSize, nBatches };
    return batchSizeAndNumber;
}

void Sequential::setDropoutOnTrain() {
    for (auto& layer : layers) {
        std::shared_ptr<Dropout> dropoutLayer = std::dynamic_pointer_cast<Dropout>(layer);
        if (dropoutLayer) {
            // Do something with the Dropout layer when training
            dropoutLayer->setTrainingMode(Dropout::DropoutMode::training);
        }
    }
}

void Sequential::setDropoutOnPredict() {
    for (auto& layer : layers) {
        std::shared_ptr<Dropout> dropoutLayer = std::dynamic_pointer_cast<Dropout>(layer);
        if (dropoutLayer) {
            // Do something with the Dropout layer when predicting
            dropoutLayer->setTrainingMode(Dropout::DropoutMode::prediction);
        }
    }
}
