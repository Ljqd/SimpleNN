#include "GradientDescent.h"

GradientDescent::GradientDescent(float lr) : learningRate(lr) {}

void GradientDescent::updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& dWeights) 
{
    weights -= learningRate * dWeights;
}

void GradientDescent::updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& dBias)
{
    bias -= learningRate * dBias;
}

std::shared_ptr<AbstractOptimizer> GradientDescent::clone() const
{
    return std::make_shared<GradientDescent>(*this);
}