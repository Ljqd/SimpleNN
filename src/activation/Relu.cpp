#include "Relu.h"

Eigen::MatrixXf Relu::forward(const Eigen::MatrixXf& input)
{
    return input.unaryExpr([](float val) { return val > 0.0f ? val : 0.0f; });
}

Eigen::MatrixXf Relu::backward(const Eigen::MatrixXf& dOutput)
{
    return dOutput.unaryExpr([](float val) { return val > 0.0f ? 1.0f : 0.0f; });
}