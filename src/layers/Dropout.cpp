#include "Dropout.h"

Dropout::Dropout(double rate) : dropoutRate(rate), mode(DropoutMode::training) {}

Eigen::MatrixXf Dropout::forward(const Eigen::MatrixXf& input)
{
    if (mode == DropoutMode::training) {
        mask = Eigen::MatrixXf::Random(input.rows(), input.cols()).unaryExpr([this](float val)
            {
                return (val < dropoutRate) ? 1.0f : 0.0f;
            });
        return input.cwiseProduct(mask);
    }
    else 
    {
        return input * (1.0 - dropoutRate);
    }
}

Eigen::MatrixXf Dropout::backward(const Eigen::MatrixXf& dOutput)
{
    if (mode == DropoutMode::training) {
        return dOutput.cwiseProduct(mask);
    }
    else 
    {
        return dOutput;
    }
}

void Dropout::updateWeights(double learningRate)
{
    // No weights to update in dropout
}

void Dropout::setTrainingMode(DropoutMode mode) {
    this->mode = mode;
}