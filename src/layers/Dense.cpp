#include "Dense.h"

#include <iostream>

Dense::Dense(int inputSize, int outputSize, std::shared_ptr<AbstractActivation> act, const std::string& initializationType)
    : activation(act)
{
    if (initializationType == "xavier") {
        float variance = sqrt(2.0 / (inputSize + outputSize));
        weights = Eigen::MatrixXf::Random(inputSize, outputSize) * variance;
    }
    else if (initializationType == "he") {  // He initialization, which is often recommended for ReLU
        float variance = sqrt(2.0 / inputSize);
        weights = Eigen::MatrixXf::Random(inputSize, outputSize) * variance;
    }
    else {
        weights = Eigen::MatrixXf::Random(inputSize, outputSize); // default
    }

    bias = Eigen::VectorXf::Zero(outputSize);
}


Eigen::MatrixXf Dense::forward(const Eigen::MatrixXf& input)
{
    assert(input.cols() == weights.rows() && "Input dimension does not match weights dimension!");

    lastInput = input;

    Z = lastInput * weights;
    for (int i = 0; i < Z.rows(); ++i) {
        Z.row(i) += bias;
    }
    A = activation->forward(Z);

    return A;
}


Eigen::MatrixXf Dense::backward(const Eigen::MatrixXf& dOutput)
{
    // 1. Calculate the gradient dZ
    Eigen::MatrixXf dZ = dOutput.array() * activation->backward(Z).array();

    int numSamples = dZ.rows();

    // 2. Compute gradients dW and dB
    dWeights = (lastInput.transpose() * dZ) / numSamples;
    dBias = dZ.colwise().sum() / numSamples;

    return dZ * weights.transpose();
}


void Dense::updateWeights()
{
    optimizer->updateWeights(weights, dWeights);
    optimizer->updateBias(bias, dBias);
}

void Dense::setOptimizer(std::shared_ptr<AbstractOptimizer> optimizer)
{
    this->optimizer = optimizer;
}

