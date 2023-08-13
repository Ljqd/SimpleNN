#pragma once

#include "AbstractOptimizer.h"

class Momentum : public AbstractOptimizer
{
public:
    Momentum(float lr, float beta = 0.9);

    void updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& dWeights) override;
    void updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& dBias) override;

    std::shared_ptr<AbstractOptimizer> clone() const override;

private:
    float learningRate;
    float beta;
    Eigen::MatrixXf velocityWeights;  // For storing the previous weight updates
    Eigen::VectorXf velocityBias;     // For storing the previous bias updates
};
