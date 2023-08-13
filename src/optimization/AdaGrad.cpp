#include "AdaGrad.h"

AdaGrad::AdaGrad(float lr, float eps)
    : learningRate(lr), epsilon(eps)
{
    // Since we don't know the size of weights and biases at this point,
    // We'll need to initialize the accumulated squared gradients when we first update the weights/biases.
}

void AdaGrad::updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& dWeights)
{
    if (accumulatedSquaredGradientsWeights.rows() == 0 && accumulatedSquaredGradientsWeights.cols() == 0)
    {
        accumulatedSquaredGradientsWeights = Eigen::MatrixXf::Zero(dWeights.rows(), dWeights.cols());
    }

    accumulatedSquaredGradientsWeights.array() += dWeights.array().square();
    weights.array() -= learningRate * dWeights.array() / (accumulatedSquaredGradientsWeights.array().sqrt() + epsilon);
}

void AdaGrad::updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& dBias)
{
    if (accumulatedSquaredGradientsBias.size() == 0)
    {
        accumulatedSquaredGradientsBias = Eigen::VectorXf::Zero(dBias.size());
    }

    accumulatedSquaredGradientsBias.array() += dBias.array().square();
    bias.array() -= learningRate * dBias.array() / (accumulatedSquaredGradientsBias.array().sqrt() + epsilon);
}

std::shared_ptr<AbstractOptimizer> AdaGrad::clone() const
{
    return std::make_shared<AdaGrad>(*this);
}