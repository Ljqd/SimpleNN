#include "Sequential.h"

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
    int batch_size)
{
    int nSamples = data.rows();

    int nBatches;
    if (batch_size > 0)
    {
        nBatches = (nSamples + batch_size - 1) / batch_size;
    }
    else
    {
        nBatches = 1;
        batch_size = nSamples;
    }

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double averageLoss = 0.0;

        for (int batch = 0; batch < nBatches; ++batch)
        {
            int startIdx = batch * batch_size;
            int endIdx = std::min(startIdx + batch_size, nSamples); // ensure we don't go out of bounds
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