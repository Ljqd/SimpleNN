#include "MeanSquareError.h"

double MeanSquareError::forward(const Eigen::MatrixXf& predicted, const Eigen::MatrixXf& actual)
{
    Eigen::MatrixXf diff = predicted - actual;
    return diff.squaredNorm() / diff.rows();
}

Eigen::MatrixXf MeanSquareError::backward(const Eigen::MatrixXf& predicted, const Eigen::MatrixXf& actual)
{
    return (2.0 / predicted.rows()) * (predicted - actual);
}