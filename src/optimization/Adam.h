#pragma once

#include "AbstractOptimizer.h"

class Adam : public AbstractOptimizer
{
public:
    Adam(float lr, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);

    void updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& dWeights) override;
    void updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& dBias) override;

    std::shared_ptr<AbstractOptimizer> clone() const override;

private:
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;

    int timestep;  // For the bias-corrected moment estimation

    Eigen::MatrixXf mWeights;
    Eigen::MatrixXf vWeights;
    Eigen::VectorXf mBias;
    Eigen::VectorXf vBias;
};
