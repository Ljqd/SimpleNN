#pragma once

#include "AbstractCost.h"

class BinaryCrossEntropy : public AbstractCost
{
public:
    double forward(const Eigen::MatrixXf& predicted, const Eigen::MatrixXf& actual) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& predicted, const Eigen::MatrixXf& actual) override;
};
