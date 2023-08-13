#pragma once

#include "AbstractOptimizer.h"

class RMSProp : public AbstractOptimizer
{
public:
    RMSProp(float lr, float decayRate = 0.9, float epsilon = 1e-8);

    void updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& dWeights) override;
    void updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& dBias) override;

    std::shared_ptr<AbstractOptimizer> clone() const override;

private:
    float learningRate;
    float decayRate;
    float epsilon;  // Small value to avoid division by zero

    Eigen::MatrixXf accumulatedSquaredGradientsWeights;
    Eigen::VectorXf accumulatedSquaredGradientsBias;
};
