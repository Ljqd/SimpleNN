#include "RMSProp.h"

RMSProp::RMSProp(float lr, float decay, float eps)
    : learningRate(lr), decayRate(decay), epsilon(eps)
{
    // Initialization of accumulatedSquaredGradients will happen on the first weight update.
}

void RMSProp::updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& dWeights)
{
    if (accumulatedSquaredGradientsWeights.rows() == 0 && accumulatedSquaredGradientsWeights.cols() == 0)
    {
        accumulatedSquaredGradientsWeights = Eigen::MatrixXf::Zero(dWeights.rows(), dWeights.cols());
    }

    accumulatedSquaredGradientsWeights = decayRate * accumulatedSquaredGradientsWeights +
        (1.0 - decayRate) * dWeights.array().square().matrix();

    weights.array() -= learningRate * dWeights.array() / (accumulatedSquaredGradientsWeights.array().sqrt() + epsilon);
}

void RMSProp::updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& dBias)
{
    if (accumulatedSquaredGradientsBias.size() == 0)
    {
        accumulatedSquaredGradientsBias = Eigen::VectorXf::Zero(dBias.size());
    }

    accumulatedSquaredGradientsBias = decayRate * accumulatedSquaredGradientsBias +
        (1.0 - decayRate) * dBias.array().square().matrix();

    bias.array() -= learningRate * dBias.array() / (accumulatedSquaredGradientsBias.array().sqrt() + epsilon);
}

std::shared_ptr<AbstractOptimizer> RMSProp::clone() const
{
    return std::make_shared<RMSProp>(*this);
}
