#include "Momentum.h"

Momentum::Momentum(float lr, float momentumTerm)
    : learningRate(lr), beta(beta)
{
    // Since we don't know the size of weights and biases at this point,
    // We'll need to initialize the velocities when we first update the weights/biases.
}

void Momentum::updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& dWeights)
{
    if (velocityWeights.rows() == 0 && velocityWeights.cols() == 0)  // Check if velocities are uninitialized
    {
        velocityWeights = Eigen::MatrixXf::Zero(dWeights.rows(), dWeights.cols());
    }

    velocityWeights = beta * velocityWeights + learningRate * dWeights;
    weights -= velocityWeights;
}

void Momentum::updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& dBias)
{
    if (velocityBias.size() == 0)  // Check if velocities are uninitialized
    {
        velocityBias = Eigen::VectorXf::Zero(dBias.size());
    }

    velocityBias = beta * velocityBias + learningRate * dBias;
    bias -= velocityBias;
}

std::shared_ptr<AbstractOptimizer> Momentum::clone() const
{
    return std::make_shared<Momentum>(*this);
}