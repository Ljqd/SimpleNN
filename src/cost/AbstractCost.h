#pragma once

#include <Eigen/Dense>

class AbstractCost
{
public:
    virtual ~AbstractCost() {}

    // Calculate the loss
    virtual double forward(const Eigen::MatrixXf& predicted, const Eigen::MatrixXf& actual) = 0;

    // Compute the gradient of the loss w.r.t. the predicted values
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& predicted, const Eigen::MatrixXf& actual) = 0;
};

