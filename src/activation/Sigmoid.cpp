#include "Sigmoid.h"

Eigen::MatrixXf Sigmoid::forward(const Eigen::MatrixXf& input)
{
    return 1.0 / (1.0 + (-input.array()).exp());
}

Eigen::MatrixXf Sigmoid::backward(const Eigen::MatrixXf& dOutput)
{
    Eigen::MatrixXf sigmoid = forward(dOutput);
    return sigmoid.array() * (1.0 - sigmoid.array());
}