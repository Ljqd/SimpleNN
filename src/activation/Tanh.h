#pragma once

#include "AbstractActivation.h"

class Tanh : public AbstractActivation
{
public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& input) override;
};
