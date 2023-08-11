#include "Tanh.h"

Eigen::MatrixXf Tanh::forward(const Eigen::MatrixXf& input)
{
    return input.array().tanh();
}

Eigen::MatrixXf Tanh::backward(const Eigen::MatrixXf& input)
{
    Eigen::MatrixXf tanhVal = forward(input);
    return (1 - tanhVal.array().square());
}