#include "BinaryCrossEntropy.h"

double BinaryCrossEntropy::forward(const Eigen::MatrixXf& predicted, const Eigen::MatrixXf& actual)
{
    // Binary Cross Entropy Loss: -[y*log(y_hat) + (1-y)*log(1-y_hat)]
    // Sum over all samples and average
    return -(actual.array() * predicted.array().log() + (1 - actual.array()) * (1 - predicted.array()).log()).sum() / predicted.rows();
}

Eigen::MatrixXf BinaryCrossEntropy::backward(const Eigen::MatrixXf& predicted, const Eigen::MatrixXf& actual)
{
    // Gradient of Binary Cross Entropy with respect to predicted values: 
    // -[y/y_hat - (1-y)/(1-y_hat)]
    return -(actual.array() / predicted.array() - (1 - actual.array()) / (1 - predicted.array()));
}