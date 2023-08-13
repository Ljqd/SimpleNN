#pragma once

#include <memory>
#include <Eigen/Dense>

class AbstractOptimizer 
{
public:
    virtual void updateWeights(Eigen::MatrixXf& weights, const Eigen::MatrixXf& dWeights) = 0;
    virtual void updateBias(Eigen::VectorXf& bias, const Eigen::VectorXf& dBias) = 0;

    virtual std::shared_ptr<AbstractOptimizer> clone() const = 0;
};