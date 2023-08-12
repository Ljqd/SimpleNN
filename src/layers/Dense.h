#pragma once

#include <string>

#include "AbstractLayer.h"
#include "../activation/AbstractActivation.h"

class Dense : public AbstractLayer
{
public:
    Dense(int inputSize, int outputSize, std::shared_ptr<AbstractActivation> act, const std::string& initializationType="default");

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& dOutput) override;

    // Weight update function for training
    void updateWeights(double learningRate) override;

private:
    Eigen::MatrixXf weights;
    Eigen::VectorXf bias;
    std::shared_ptr<AbstractActivation> activation;

    Eigen::MatrixXf lastInput;
    Eigen::MatrixXf Z;
    Eigen::MatrixXf A;

    Eigen::MatrixXf dWeights;
    Eigen::VectorXf dBias;
};