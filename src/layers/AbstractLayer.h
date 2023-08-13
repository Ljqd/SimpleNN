#pragma once

#include <Eigen/Dense>
#include <memory>

#include "../optimization/AbstractOptimizer.h"


class AbstractLayer
{
public:
    virtual ~AbstractLayer() {}

    // Forward pass function
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input) = 0;

    // Backward pass function
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& dOutput) = 0;

    // Weight update function for training
    virtual void updateWeights() = 0;

    // Define the way how you will update your weights
    // the beauty - code in Sequential almost don't change, we just don't care what happens within updateWeights
    virtual void setOptimizer(std::shared_ptr<AbstractOptimizer> optimizer) = 0;
};