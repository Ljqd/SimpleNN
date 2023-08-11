#pragma once

#include <Eigen/Dense>

class AbstractActivation
{
public:
    virtual ~AbstractActivation() {}
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input) = 0;
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& dOutput) = 0;
};