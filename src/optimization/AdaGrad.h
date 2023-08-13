#pragma once

#include "AbstractOptimizer.h"

class AdaGrad : public AbstractOptimizer
{
public:
    AdaGrad(float lr, float epsilon = 1e-8);

    void updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& dWeights) override;
    void updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& dBias) override;

    std::shared_ptr<AbstractOptimizer> clone() const override;

private:
    float learningRate;
    float epsilon;  // Small value to avoid division by zero
    Eigen::MatrixXf accumulatedSquaredGradientsWeights;
    Eigen::VectorXf accumulatedSquaredGradientsBias;
};
