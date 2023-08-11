#pragma once

#include <Eigen/Dense>

class AbstractLayer
{
public:
    virtual ~AbstractLayer() {}

    // Forward pass function
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input) = 0;

    // Backward pass function
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& dOutput) = 0;

    // Weight update function for training
    virtual void updateWeights(double learningRate) = 0;
};