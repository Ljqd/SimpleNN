#include "Adam.h"

Adam::Adam(float lr, float b1, float b2, float eps)
    : learningRate(lr), beta1(b1), beta2(b2), epsilon(eps), timestep(1)
{
    // Initialization of mWeights, vWeights, mBias, and vBias will happen on the first weight update.
}

void Adam::updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& dWeights)
{
    if (mWeights.rows() == 0 && mWeights.cols() == 0)
    {
        mWeights = Eigen::MatrixXf::Zero(dWeights.rows(), dWeights.cols());
        vWeights = Eigen::MatrixXf::Zero(dWeights.rows(), dWeights.cols());
    }

    mWeights = beta1 * mWeights + (1.0f - beta1) * dWeights;
    vWeights = beta2 * vWeights + (1.0f - beta2) * dWeights.array().square().matrix();

    Eigen::MatrixXf mWeightsCorrected = mWeights / (1.0f - std::pow(beta1, timestep));
    Eigen::MatrixXf vWeightsCorrected = vWeights / (1.0f - std::pow(beta2, timestep));

    weights.array() -= learningRate * mWeightsCorrected.array() / (vWeightsCorrected.array().sqrt() + epsilon);
}

void Adam::updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& dBias)
{
    if (mBias.size() == 0)
    {
        mBias = Eigen::VectorXf::Zero(dBias.size());
        vBias = Eigen::VectorXf::Zero(dBias.size());
    }

    mBias = beta1 * mBias + (1.0f - beta1) * dBias;
    vBias = beta2 * vBias + (1.0f - beta2) * dBias.array().square().matrix();

    Eigen::VectorXf mBiasCorrected = mBias / (1.0f - std::pow(beta1, timestep));
    Eigen::VectorXf vBiasCorrected = vBias / (1.0f - std::pow(beta2, timestep));

    bias.array() -= learningRate * mBiasCorrected.array() / (vBiasCorrected.array().sqrt() + epsilon);
}

std::shared_ptr<AbstractOptimizer> Adam::clone() const
{
    return std::make_shared<Adam>(*this);
}
