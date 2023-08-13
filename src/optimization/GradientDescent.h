#pragma once

#include "AbstractOptimizer.h"

class GradientDescent : public AbstractOptimizer 
{
public:
    GradientDescent(float lr);

    void updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& dWeights) override;
    void updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& dBias) override;

    std::shared_ptr<AbstractOptimizer> clone() const override;
private:
    float learningRate;
};